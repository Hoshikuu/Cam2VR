#include <sstream>
#include <stdexcept>
#include <utility>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <chrono>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <exception>
#include <condition_variable>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include <Windows.h>
#include <dshow.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>

#include <turbojpeg.h>

#pragma comment(lib, "Mfplat.lib")
#pragma comment(lib, "Mf.lib")
#pragma comment(lib, "Mfreadwrite.lib")
#pragma comment(lib, "Mfuuid.lib")
#pragma comment(lib, "Ole32.lib")
#pragma comment(lib, "Strmiids.lib")
#pragma comment(lib, "turbojpeg.lib")

#pragma pack(push, 1)

struct BitmapFileHeader
{
    std::uint16_t type = 0x4D42;
    std::uint32_t fileSize = 0;
    std::uint16_t reserved1 = 0;
    std::uint16_t reserved2 = 0;
    std::uint32_t dataOffset = 54;
};

struct BitmapInfoHeader
{
    std::uint32_t headerSize = 40;
    std::int32_t width = 0;
    std::int32_t height = 0;
    std::uint16_t planes = 1;
    std::uint16_t bitsPerPixel = 24;
    std::uint32_t compression = 0;
    std::uint32_t imageSize = 0;
    std::int32_t xPixelsPerMeter = 2835;
    std::int32_t yPixelsPerMeter = 2835;
    std::uint32_t colorsUsed = 0;
    std::uint32_t importantColors = 0;
};

#pragma pack(pop)

struct RGBFrame
{
    int width = 0;
    int height = 0;
    LONGLONG timestamp = 0;

    std::vector<std::uint8_t> pixels;
};

namespace
{
    const wchar_t* VideoFormatToString(const GUID& subtype)
    {
        if (IsEqualGUID(subtype, MFVideoFormat_YUY2))
        {
            return L"YUY2";
        }

        if (IsEqualGUID(subtype, MFVideoFormat_MJPG))
        {
            return L"MJPG";
        }

        if (IsEqualGUID(subtype, MFVideoFormat_NV12))
        {
            return L"NV12";
        }

        if (IsEqualGUID(subtype, MFVideoFormat_RGB24))
        {
            return L"RGB24";
        }

        if (IsEqualGUID(subtype, MFVideoFormat_RGB32))
        {
            return L"RGB32";
        }

        return L"UNKNOWN";
    }

    void PrintHResult(const char* message, HRESULT hr)
    {
        std::cout
            << message
            << " HRESULT: 0x"
            << std::hex
            << static_cast<unsigned long>(hr)
            << std::dec
            << '\n';
    }

    void PrintMediaTypeInfo(IMFMediaType* mediaType, DWORD index)
    {
        if (!mediaType)
        {
            return;
        }

        UINT32 width = 0;
        UINT32 height = 0;

        HRESULT hr = MFGetAttributeSize(
            mediaType,
            MF_MT_FRAME_SIZE,
            &width,
            &height
        );

        if (FAILED(hr))
        {
            PrintHResult("No se pudo leer la resolucion.", hr);
            return;
        }

        UINT32 fpsNumerator = 0;
        UINT32 fpsDenominator = 0;

        hr = MFGetAttributeRatio(
            mediaType,
            MF_MT_FRAME_RATE,
            &fpsNumerator,
            &fpsDenominator
        );

        if (FAILED(hr))
        {
            PrintHResult("No se pudo leer la tasa de frames.", hr);
            return;
        }

        GUID subtype{};

        hr = mediaType->GetGUID(
            MF_MT_SUBTYPE,
            &subtype
        );

        if (FAILED(hr))
        {
            PrintHResult("No se pudo leer el formato de video.", hr);
            return;
        }

        double fps = 0.0;

        if (fpsDenominator != 0)
        {
            fps =
                static_cast<double>(fpsNumerator) /
                static_cast<double>(fpsDenominator);
        }

        std::wcout
            << L"[" << index << L"] "
            << width << L"x" << height
            << L" @ " << fps << L" FPS"
            << L" - "
            << VideoFormatToString(subtype)
            << L'\n';
    }

    void ReleaseDeviceArray(IMFActivate** devices, UINT32 deviceCount)
    {
        if (!devices)
        {
            return;
        }

        for (UINT32 i = 0; i < deviceCount; i++)
        {
            if (devices[i])
            {
                devices[i]->Release();
            }
        }

        CoTaskMemFree(devices);
    }
}

bool SaveRGBFrameAsBMP(const RGBFrame& frame, const char* filename)
{
    if (
        frame.width <= 0 ||
        frame.height <= 0 ||
        frame.pixels.empty()
        )
    {
        std::cout << "No hay una imagen RGB valida para guardar\n";
        return false;
    }

    const std::size_t expectedSize =
        static_cast<std::size_t>(frame.width) *
        static_cast<std::size_t>(frame.height) *
        3;

    if (frame.pixels.size() < expectedSize)
    {
        std::cout << "El buffer RGB es mas pequeno de lo esperado\n";
        return false;
    }

    const int rowBytes = frame.width * 3;
    const int padding = (4 - (rowBytes % 4)) % 4;
    const int storedRowBytes = rowBytes + padding;

    const std::uint32_t imageSize =
        static_cast<std::uint32_t>(
            storedRowBytes * frame.height
            );

    BitmapFileHeader fileHeader;
    BitmapInfoHeader infoHeader;

    fileHeader.fileSize =
        fileHeader.dataOffset + imageSize;

    infoHeader.width = frame.width;
    infoHeader.height = frame.height;
    infoHeader.imageSize = imageSize;

    std::ofstream file(filename, std::ios::binary);

    if (!file)
    {
        std::cout << "No se pudo crear el archivo BMP\n";
        return false;
    }

    file.write(
        reinterpret_cast<const char*>(&fileHeader),
        sizeof(fileHeader)
    );

    file.write(
        reinterpret_cast<const char*>(&infoHeader),
        sizeof(infoHeader)
    );

    const unsigned char paddingBytes[3] = { 0, 0, 0 };

    // BMP guarda las filas de abajo hacia arriba y usa BGR.
    for (int y = frame.height - 1; y >= 0; y--)
    {
        const std::uint8_t* row =
            frame.pixels.data() +
            static_cast<std::size_t>(y) *
            static_cast<std::size_t>(frame.width) *
            3;

        for (int x = 0; x < frame.width; x++)
        {
            const unsigned char red = row[x * 3 + 0];
            const unsigned char green = row[x * 3 + 1];
            const unsigned char blue = row[x * 3 + 2];

            const unsigned char bgr[3] =
            {
                blue,
                green,
                red
            };

            file.write(
                reinterpret_cast<const char*>(bgr),
                3
            );
        }

        file.write(
            reinterpret_cast<const char*>(paddingBytes),
            padding
        );
    }

    if (!file)
    {
        std::cout << "Error escribiendo el archivo BMP\n";
        return false;
    }

    std::cout
        << "Frame guardado en: "
        << filename
        << '\n';

    return true;
}

class Camera
{
public:
    Camera() = default;

    ~Camera()
    {
        Close();
    }

    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    bool Open(
        UINT32 cameraIndex,
        DWORD formatIndex,
        long manualExposure = -6,
        bool listFormats = false
    )
    {
        Close();

        if (!OpenMediaSource(cameraIndex))
        {
            Close();
            return false;
        }

        if (!ConfigureExposure(manualExposure))
        {
            std::cout
                << "Aviso: se continuara usando la exposicion del driver\n";
        }

        if (!CreateSourceReader())
        {
            Close();
            return false;
        }

        if (listFormats)
        {
            ListFormats();
        }

        if (!SetFormatByIndex(formatIndex))
        {
            Close();
            return false;
        }

        decoder_ = tj3Init(TJINIT_DECOMPRESS);

        if (!decoder_)
        {
            std::cout << "No se pudo crear el decodificador MJPEG\n";
            Close();
            return false;
        }

        isOpen_ = true;

        std::cout << "Camara completamente configurada\n";

        return true;
    }

    bool StartCapture()
    {
        if (!isOpen_ || !sourceReader_ || !decoder_)
        {
            std::cout
                << "No se puede iniciar la captura: "
                << "la camara no esta preparada\n";

            return false;
        }

        if (captureRunning_.load())
        {
            // Ya existe un hilo capturando.
            return true;
        }

        /*
            Si un hilo anterior termino por un error, el objeto
            std::thread puede seguir siendo joinable. Lo recogemos
            antes de crear otro hilo.
        */
        if (captureThread_.joinable())
        {
            captureThread_.join();
        }

        capturedFrameCount_.store(0);

        {
            std::lock_guard<std::mutex> lock(frameMutex_);

            latestFrame_ = RGBFrame{};
            latestFrameSequence_ = 0;
            hasLatestFrame_ = false;
        }

        // Elimina samples antiguos antes de arrancar el hilo.
        sourceReader_->Flush(
            MF_SOURCE_READER_FIRST_VIDEO_STREAM
        );

        captureRunning_.store(true);

        try
        {
            captureThread_ = std::thread(
                &Camera::CaptureLoop,
                this
            );
        }
        catch (const std::exception& exception)
        {
            captureRunning_.store(false);

            std::cout
                << "No se pudo crear el hilo de captura: "
                << exception.what()
                << '\n';

            return false;
        }

        std::cout << "Captura continua iniciada\n";
        return true;
    }

    void StopCapture()
    {
        captureRunning_.store(false);

        /*
            Despierta a cualquier consumidor que este esperando
            un frame nuevo mediante WaitForNextFrame().
        */
        frameCondition_.notify_all();

        /*
            ReadSample() es sincronico. A 30 FPS, join() puede
            esperar aproximadamente hasta que llegue el siguiente
            frame y el bucle compruebe captureRunning_ otra vez.
        */
        if (captureThread_.joinable())
        {
            captureThread_.join();
        }
    }

    bool GetLatestFrame(
        RGBFrame& outputFrame,
        std::uint64_t* sequence = nullptr
    ) const
    {
        std::lock_guard<std::mutex> lock(frameMutex_);

        if (!hasLatestFrame_)
        {
            return false;
        }

        /*
            Por ahora hacemos una copia segura del ultimo frame.
            En el capitulo de pybind11 estudiaremos como reducir
            esta copia al entregar la memoria a NumPy.
        */
        outputFrame = latestFrame_;

        if (sequence)
        {
            *sequence = latestFrameSequence_;
        }

        return true;
    }

    bool WaitForNextFrame(
        RGBFrame& outputFrame,
        std::uint64_t lastSequence,
        std::uint64_t& outputSequence,
        int timeoutMilliseconds = 1000
    )
    {
        std::unique_lock<std::mutex> lock(frameMutex_);

        const auto predicate =
            [this, lastSequence]()
            {
                return
                    (
                        hasLatestFrame_ &&
                        latestFrameSequence_ > lastSequence
                        )
                    ||
                    !captureRunning_.load();
            };

        bool conditionReached = false;

        if (timeoutMilliseconds < 0)
        {
            // Un valor negativo espera indefinidamente.
            frameCondition_.wait(lock, predicate);
            conditionReached = true;
        }
        else
        {
            conditionReached = frameCondition_.wait_for(
                lock,
                std::chrono::milliseconds(timeoutMilliseconds),
                predicate
            );
        }

        if (!conditionReached)
        {
            // Se agoto el tiempo de espera.
            return false;
        }

        if (
            !hasLatestFrame_ ||
            latestFrameSequence_ <= lastSequence
            )
        {
            // La captura termino sin producir un frame posterior.
            return false;
        }

        /*
            Copia segura para que el consumidor pueda usar el frame
            aunque el hilo de captura publique otro inmediatamente.
        */
        outputFrame = latestFrame_;
        outputSequence = latestFrameSequence_;

        return true;
    }

    bool IsCapturing() const noexcept
    {
        return captureRunning_.load();
    }

    std::uint64_t CapturedFrameCount() const noexcept
    {
        return capturedFrameCount_.load();
    }

private:
    bool ReadFrameBlocking(RGBFrame& outputFrame)
    {
        if (!isOpen_ || !sourceReader_ || !decoder_)
        {
            std::cout << "La camara no esta abierta\n";
            return false;
        }

        while (true)
        {
            DWORD streamIndex = 0;
            DWORD flags = 0;
            LONGLONG timestamp = 0;
            IMFSample* sample = nullptr;

            HRESULT hr = sourceReader_->ReadSample(
                MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                0,
                &streamIndex,
                &flags,
                &timestamp,
                &sample
            );

            if (FAILED(hr))
            {
                PrintHResult("Error leyendo el frame.", hr);
                return false;
            }

            if (flags & MF_SOURCE_READERF_ERROR)
            {
                if (sample)
                {
                    sample->Release();
                }

                std::cout << "El SourceReader notifico un error\n";
                return false;
            }

            if (flags & MF_SOURCE_READERF_ENDOFSTREAM)
            {
                if (sample)
                {
                    sample->Release();
                }

                std::cout << "Fin del stream\n";
                return false;
            }

            if (
                flags &
                (MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED |
                    MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED)
                )
            {
                if (!UpdateCurrentFormat())
                {
                    if (sample)
                    {
                        sample->Release();
                    }

                    return false;
                }
            }

            if (
                (flags & MF_SOURCE_READERF_STREAMTICK) ||
                !sample
                )
            {
                if (sample)
                {
                    sample->Release();
                }

                continue;
            }

            IMFMediaBuffer* buffer = nullptr;

            hr = sample->ConvertToContiguousBuffer(&buffer);

            if (FAILED(hr))
            {
                PrintHResult(
                    "No se pudo obtener el buffer continuo.",
                    hr
                );

                sample->Release();
                return false;
            }

            BYTE* data = nullptr;
            DWORD maximumLength = 0;
            DWORD currentLength = 0;

            hr = buffer->Lock(
                &data,
                &maximumLength,
                &currentLength
            );

            if (FAILED(hr))
            {
                PrintHResult("No se pudo bloquear el buffer.", hr);

                buffer->Release();
                sample->Release();
                return false;
            }

            const bool decodeOK = DecodeMJPEGToRGB(
                data,
                static_cast<std::size_t>(currentLength),
                outputFrame
            );

            outputFrame.timestamp = timestamp;

            const HRESULT unlockHr = buffer->Unlock();

            buffer->Release();
            sample->Release();

            if (FAILED(unlockHr))
            {
                PrintHResult("No se pudo desbloquear el buffer.", unlockHr);
                return false;
            }

            return decodeOK;
        }
    }

public:
    void Close()
    {
        // El hilo debe terminar antes de liberar SourceReader y MediaSource.
        StopCapture();

        isOpen_ = false;

        if (sourceReader_)
        {
            sourceReader_->Flush(
                MF_SOURCE_READER_FIRST_VIDEO_STREAM
            );

            sourceReader_->Release();
            sourceReader_ = nullptr;
        }

        if (mediaSource_)
        {
            mediaSource_->Shutdown();
            mediaSource_->Release();
            mediaSource_ = nullptr;
        }

        if (decoder_)
        {
            tj3Destroy(decoder_);
            decoder_ = nullptr;
        }

        width_ = 0;
        height_ = 0;
        subtype_ = GUID_NULL;

        {
            std::lock_guard<std::mutex> lock(frameMutex_);

            latestFrame_ = RGBFrame{};
            latestFrameSequence_ = 0;
            hasLatestFrame_ = false;
        }

        frameCondition_.notify_all();
    }

    bool IsOpen() const noexcept
    {
        return isOpen_;
    }

    UINT32 Width() const noexcept
    {
        return width_;
    }

    UINT32 Height() const noexcept
    {
        return height_;
    }

    void ListFormats()
    {
        if (!sourceReader_)
        {
            std::cout << "No existe SourceReader\n";
            return;
        }

        std::cout << "\nFormatos de la camara:\n";

        DWORD index = 0;

        while (true)
        {
            IMFMediaType* mediaType = nullptr;

            HRESULT hr = sourceReader_->GetNativeMediaType(
                MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                index,
                &mediaType
            );

            if (FAILED(hr))
            {
                break;
            }

            PrintMediaTypeInfo(mediaType, index);

            mediaType->Release();
            index++;
        }
    }

private:
    bool OpenMediaSource(UINT32 cameraIndex)
    {
        IMFAttributes* attributes = nullptr;
        IMFActivate** devices = nullptr;
        UINT32 deviceCount = 0;

        HRESULT hr = MFCreateAttributes(
            &attributes,
            1
        );

        if (FAILED(hr))
        {
            PrintHResult("No se pudieron crear los atributos.", hr);
            return false;
        }

        hr = attributes->SetGUID(
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID
        );

        if (FAILED(hr))
        {
            PrintHResult("No se pudo configurar el filtro.", hr);
            attributes->Release();
            return false;
        }

        hr = MFEnumDeviceSources(
            attributes,
            &devices,
            &deviceCount
        );

        attributes->Release();
        attributes = nullptr;

        if (FAILED(hr))
        {
            PrintHResult("No se pudieron enumerar las camaras.", hr);
            return false;
        }

        if (!devices || deviceCount == 0)
        {
            std::cout << "No se encontraron camaras\n";
            ReleaseDeviceArray(devices, deviceCount);
            return false;
        }

        std::cout << "\nCamaras disponibles:\n";

        for (UINT32 i = 0; i < deviceCount; i++)
        {
            WCHAR* friendlyName = nullptr;
            UINT32 nameLength = 0;

            hr = devices[i]->GetAllocatedString(
                MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,
                &friendlyName,
                &nameLength
            );

            if (SUCCEEDED(hr) && friendlyName)
            {
                std::wcout
                    << L"[" << i << L"] "
                    << friendlyName
                    << L'\n';

                CoTaskMemFree(friendlyName);
            }
        }

        if (cameraIndex >= deviceCount)
        {
            std::cout
                << "Indice de camara no valido: "
                << cameraIndex
                << '\n';

            ReleaseDeviceArray(devices, deviceCount);
            return false;
        }

        hr = devices[cameraIndex]->ActivateObject(
            IID_PPV_ARGS(&mediaSource_)
        );

        ReleaseDeviceArray(devices, deviceCount);

        if (FAILED(hr))
        {
            PrintHResult("No se pudo abrir la camara.", hr);
            mediaSource_ = nullptr;
            return false;
        }

        std::cout
            << "Camara "
            << cameraIndex
            << " abierta correctamente\n";

        return true;
    }

    bool ConfigureExposure(long requestedExposure)
    {
        if (!mediaSource_)
        {
            return false;
        }

        IAMCameraControl* cameraControl = nullptr;

        HRESULT hr = mediaSource_->QueryInterface(
            IID_PPV_ARGS(&cameraControl)
        );

        if (FAILED(hr) || !cameraControl)
        {
            PrintHResult(
                "No se pudo obtener IAMCameraControl.",
                hr
            );

            return false;
        }

        long minimum = 0;
        long maximum = 0;
        long step = 0;
        long defaultValue = 0;
        long capabilities = 0;

        hr = cameraControl->GetRange(
            CameraControl_Exposure,
            &minimum,
            &maximum,
            &step,
            &defaultValue,
            &capabilities
        );

        if (FAILED(hr))
        {
            PrintHResult(
                "No se pudo obtener el rango de exposicion.",
                hr
            );

            cameraControl->Release();
            return false;
        }

        std::cout
            << "Rango de exposicion: "
            << minimum << " - " << maximum
            << " | Paso: " << step
            << " | Defecto: " << defaultValue
            << '\n';

        if (!(capabilities & CameraControl_Flags_Manual))
        {
            std::cout
                << "La camara no soporta exposicion manual\n";

            cameraControl->Release();
            return false;
        }

        long exposure = std::clamp(
            requestedExposure,
            minimum,
            maximum
        );

        if (step > 0)
        {
            exposure =
                minimum +
                ((exposure - minimum) / step) * step;
        }

        hr = cameraControl->Set(
            CameraControl_Exposure,
            exposure,
            CameraControl_Flags_Manual
        );

        if (FAILED(hr))
        {
            PrintHResult(
                "No se pudo establecer la exposicion manual.",
                hr
            );

            cameraControl->Release();
            return false;
        }

        long currentExposure = 0;
        long currentFlags = 0;

        hr = cameraControl->Get(
            CameraControl_Exposure,
            &currentExposure,
            &currentFlags
        );

        if (SUCCEEDED(hr))
        {
            std::cout
                << "Exposicion actual: "
                << currentExposure
                << " | Modo: "
                << (
                    currentFlags & CameraControl_Flags_Manual
                    ? "Manual"
                    : "Automatico"
                    )
                << '\n';
        }

        cameraControl->Release();
        return true;
    }

    bool CreateSourceReader()
    {
        if (!mediaSource_)
        {
            return false;
        }

        IMFAttributes* readerAttributes = nullptr;

        HRESULT hr = MFCreateAttributes(
            &readerAttributes,
            3
        );

        if (FAILED(hr))
        {
            PrintHResult(
                "No se pudieron crear los atributos del SourceReader.",
                hr
            );

            return false;
        }

        hr = readerAttributes->SetUINT32(
            MF_LOW_LATENCY,
            TRUE
        );

        if (SUCCEEDED(hr))
        {
            hr = readerAttributes->SetUINT32(
                MF_READWRITE_DISABLE_CONVERTERS,
                TRUE
            );
        }

        if (SUCCEEDED(hr))
        {
            hr = readerAttributes->SetUINT32(
                MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING,
                FALSE
            );
        }

        if (FAILED(hr))
        {
            PrintHResult(
                "No se pudieron configurar los atributos del SourceReader.",
                hr
            );

            readerAttributes->Release();
            return false;
        }

        hr = MFCreateSourceReaderFromMediaSource(
            mediaSource_,
            readerAttributes,
            &sourceReader_
        );

        readerAttributes->Release();
        readerAttributes = nullptr;

        if (FAILED(hr))
        {
            PrintHResult("No se pudo crear el SourceReader.", hr);
            sourceReader_ = nullptr;
            return false;
        }

        hr = sourceReader_->SetStreamSelection(
            MF_SOURCE_READER_ALL_STREAMS,
            FALSE
        );

        if (FAILED(hr))
        {
            PrintHResult("No se pudieron desactivar los streams.", hr);
            return false;
        }

        hr = sourceReader_->SetStreamSelection(
            MF_SOURCE_READER_FIRST_VIDEO_STREAM,
            TRUE
        );

        if (FAILED(hr))
        {
            PrintHResult("No se pudo activar el stream de video.", hr);
            return false;
        }

        std::cout << "SourceReader creado correctamente\n";
        return true;
    }

    bool SetFormatByIndex(DWORD formatIndex)
    {
        if (!sourceReader_)
        {
            return false;
        }

        IMFMediaType* mediaType = nullptr;

        HRESULT hr = sourceReader_->GetNativeMediaType(
            MF_SOURCE_READER_FIRST_VIDEO_STREAM,
            formatIndex,
            &mediaType
        );

        if (FAILED(hr))
        {
            PrintHResult(
                "No se pudo obtener el formato seleccionado.",
                hr
            );

            return false;
        }

        std::cout << "\nFormato seleccionado:\n";
        PrintMediaTypeInfo(mediaType, formatIndex);

        hr = sourceReader_->SetCurrentMediaType(
            MF_SOURCE_READER_FIRST_VIDEO_STREAM,
            nullptr,
            mediaType
        );

        mediaType->Release();
        mediaType = nullptr;

        if (FAILED(hr))
        {
            PrintHResult(
                "No se pudo aplicar el formato seleccionado.",
                hr
            );

            return false;
        }

        if (!UpdateCurrentFormat())
        {
            return false;
        }

        if (!IsEqualGUID(subtype_, MFVideoFormat_MJPG))
        {
            std::wcout
                << L"El formato activo es "
                << VideoFormatToString(subtype_)
                << L", pero el decodificador necesita MJPG\n";

            return false;
        }

        hr = sourceReader_->Flush(
            MF_SOURCE_READER_FIRST_VIDEO_STREAM
        );

        if (FAILED(hr))
        {
            PrintHResult(
                "Aviso: no se pudo vaciar el SourceReader.",
                hr
            );
        }

        return true;
    }

    bool UpdateCurrentFormat()
    {
        if (!sourceReader_)
        {
            return false;
        }

        IMFMediaType* currentType = nullptr;

        HRESULT hr = sourceReader_->GetCurrentMediaType(
            MF_SOURCE_READER_FIRST_VIDEO_STREAM,
            &currentType
        );

        if (FAILED(hr))
        {
            PrintHResult(
                "No se pudo obtener el formato actual.",
                hr
            );

            return false;
        }

        hr = MFGetAttributeSize(
            currentType,
            MF_MT_FRAME_SIZE,
            &width_,
            &height_
        );

        if (SUCCEEDED(hr))
        {
            hr = currentType->GetGUID(
                MF_MT_SUBTYPE,
                &subtype_
            );
        }

        if (SUCCEEDED(hr))
        {
            std::wcout
                << L"Formato activo: "
                << width_ << L"x" << height_
                << L" - "
                << VideoFormatToString(subtype_)
                << L'\n';
        }

        currentType->Release();

        if (FAILED(hr))
        {
            PrintHResult("No se pudo leer el formato activo.", hr);
            return false;
        }

        return true;
    }

    bool DecodeMJPEGToRGB(
        const BYTE* jpegData,
        std::size_t jpegSize,
        RGBFrame& outputFrame
    )
    {
        if (
            !decoder_ ||
            !jpegData ||
            jpegSize == 0
            )
        {
            return false;
        }

        int result = tj3DecompressHeader(
            decoder_,
            jpegData,
            jpegSize
        );

        if (result == -1)
        {
            std::cout
                << "Error al leer la cabecera JPEG: "
                << tj3GetErrorStr(decoder_)
                << '\n';

            return false;
        }

        const int width = tj3Get(
            decoder_,
            TJPARAM_JPEGWIDTH
        );

        const int height = tj3Get(
            decoder_,
            TJPARAM_JPEGHEIGHT
        );

        if (width <= 0 || height <= 0)
        {
            std::cout << "Dimensiones JPEG no validas\n";
            return false;
        }

        constexpr std::size_t channels = 3;

        const std::size_t requiredSize =
            static_cast<std::size_t>(width) *
            static_cast<std::size_t>(height) *
            channels;

        outputFrame.pixels.resize(requiredSize);

        result = tj3Decompress8(
            decoder_,
            jpegData,
            jpegSize,
            outputFrame.pixels.data(),
            0,
            TJPF_RGB
        );

        if (result == -1)
        {
            std::cout
                << "Error al decodificar el JPEG: "
                << tj3GetErrorStr(decoder_)
                << '\n';

            return false;
        }

        outputFrame.width = width;
        outputFrame.height = height;

        return true;
    }

    void CaptureLoop()
    {
        /*
            Cada hilo que utiliza COM debe inicializar COM.
            Este hilo pertenece al modelo multithreaded apartment.
        */
        const HRESULT comHr = CoInitializeEx(
            nullptr,
            COINIT_MULTITHREADED
        );

        const bool mustUninitializeCOM = SUCCEEDED(comHr);

        if (FAILED(comHr) && comHr != RPC_E_CHANGED_MODE)
        {
            PrintHResult(
                "No se pudo iniciar COM en el hilo de captura.",
                comHr
            );

            captureRunning_.store(false);
            frameCondition_.notify_all();
            return;
        }

        RGBFrame capturedFrame;

        while (captureRunning_.load())
        {
            if (!ReadFrameBlocking(capturedFrame))
            {
                if (captureRunning_.load())
                {
                    std::cout << "El hilo no pudo leer el siguiente frame\n";
                }

                break;
            }

            {
                std::lock_guard<std::mutex> lock(frameMutex_);

                latestFrame_.width = capturedFrame.width;
                latestFrame_.height = capturedFrame.height;
                latestFrame_.timestamp = capturedFrame.timestamp;

                /*
                    swap() intercambia los vectores sin copiar los
                    2.764.800 bytes de un frame 1280x720 RGB.

                    capturedFrame recibe el buffer anterior y lo
                    reutiliza durante la siguiente decodificacion.
                */
                latestFrame_.pixels.swap(
                    capturedFrame.pixels
                );

                latestFrameSequence_++;
                hasLatestFrame_ = true;
            }

            // Despierta a los consumidores que esperan un frame nuevo.
            frameCondition_.notify_all();

            capturedFrameCount_.fetch_add(1);
        }

        captureRunning_.store(false);

        // Despierta a quienes esperaban si el hilo termino o fallo.
        frameCondition_.notify_all();

        if (mustUninitializeCOM)
        {
            CoUninitialize();
        }
    }

private:
    IMFMediaSource* mediaSource_ = nullptr;
    IMFSourceReader* sourceReader_ = nullptr;
    tjhandle decoder_ = nullptr;

    std::thread captureThread_;
    std::atomic<bool> captureRunning_{ false };
    std::atomic<std::uint64_t> capturedFrameCount_{ 0 };

    mutable std::mutex frameMutex_;
    std::condition_variable frameCondition_;

    RGBFrame latestFrame_;
    std::uint64_t latestFrameSequence_ = 0;
    bool hasLatestFrame_ = false;

    UINT32 width_ = 0;
    UINT32 height_ = 0;
    GUID subtype_{};

    bool isOpen_ = false;
};

py::array_t<std::uint8_t> RGBFrameToNumpy(
    RGBFrame&& frame
)
{
    if (
        frame.width <= 0 ||
        frame.height <= 0 ||
        frame.pixels.empty()
        )
    {
        throw py::value_error(
            "El frame RGB no es valido"
        );
    }

    constexpr py::ssize_t channels = 3;

    const py::ssize_t width =
        static_cast<py::ssize_t>(
            frame.width
            );

    const py::ssize_t height =
        static_cast<py::ssize_t>(
            frame.height
            );

    const std::size_t expectedSize =
        static_cast<std::size_t>(width) *
        static_cast<std::size_t>(height) *
        static_cast<std::size_t>(channels);

    if (frame.pixels.size() < expectedSize)
    {
        throw py::value_error(
            "El buffer RGB es mas pequeno de lo esperado"
        );
    }

    /*
        Movemos el frame a memoria dinamica.

        NumPy usara directamente el vector pixels
        contenido dentro de este objeto.
    */
    auto* ownedFrame =
        new RGBFrame(
            std::move(frame)
        );

    /*
        La capsula se encargara de eliminar el
        RGBFrame cuando NumPy deje de utilizarlo.
    */
    py::capsule owner(
        ownedFrame,
        [](void* pointer)
        {
            auto* framePointer =
                static_cast<RGBFrame*>(
                    pointer
                    );

            delete framePointer;
        }
    );

    /*
        Dimensiones del ndarray:

        altura, anchura, canales
    */
    const std::array<py::ssize_t, 3> shape =
    {
        height,
        width,
        channels
    };

    /*
        Distancia en bytes para avanzar:

        - Una fila: width * 3
        - Un pixel: 3
        - Un canal: 1
    */
    const std::array<py::ssize_t, 3> strides =
    {
        width * channels *
            static_cast<py::ssize_t>(
                sizeof(std::uint8_t)
            ),

        channels *
            static_cast<py::ssize_t>(
                sizeof(std::uint8_t)
            ),

        static_cast<py::ssize_t>(
            sizeof(std::uint8_t)
        )
    };

    /*
        Construimos explicitamente los tipos que
        espera pybind11 para evitar la ambigüedad
        de las initializer lists en MSVC.
    */
    py::array::ShapeContainer numpyShape(
        shape
    );

    py::array::StridesContainer numpyStrides(
        strides
    );

    return py::array_t<std::uint8_t>(
        std::move(numpyShape),
        std::move(numpyStrides),
        ownedFrame->pixels.data(),
        owner
    );
}

class PythonCamera
{
public:
    PythonCamera()
    {
        HRESULT hr = CoInitializeEx(
            nullptr,
            COINIT_MULTITHREADED
        );

        /*
            RPC_E_CHANGED_MODE significa que Python
            o alguna librería ya inicializó COM con
            otro modelo en este hilo.

            COM sigue estando inicializado, pero no
            debemos ejecutar CoUninitialize nosotros.
        */
        if (hr == RPC_E_CHANGED_MODE)
        {
            ownsCOM_ = false;
        }
        else if (FAILED(hr))
        {
            throw std::runtime_error(
                "No se pudo inicializar COM"
            );
        }
        else
        {
            /*
                Incluye tanto S_OK como S_FALSE.

                Si CoInitializeEx tuvo éxito,
                debemos equilibrarlo después con
                CoUninitialize().
            */
            ownsCOM_ = true;
        }

        hr = MFStartup(
            MF_VERSION,
            MFSTARTUP_FULL
        );

        if (FAILED(hr))
        {
            if (ownsCOM_)
            {
                CoUninitialize();
                ownsCOM_ = false;
            }

            throw std::runtime_error(
                "No se pudo inicializar Media Foundation"
            );
        }

        mediaFoundationStarted_ = true;
    }


    ~PythonCamera()
    {
        ShutdownRuntime();
    }


    PythonCamera(
        const PythonCamera&
    ) = delete;


    PythonCamera& operator=(
        const PythonCamera&
        ) = delete;


    bool Open(
        std::uint32_t cameraIndex,
        std::uint32_t formatIndex,
        long exposure = -6,
        bool listFormats = false
    )
    {
        /*
            Open puede tardar un poco.

            Liberamos el GIL para no bloquear otros
            hilos de Python durante la operación.
        */
        py::gil_scoped_release release;

        return camera_.Open(
            static_cast<UINT32>(cameraIndex),
            static_cast<DWORD>(formatIndex),
            exposure,
            listFormats
        );
    }


    bool Start()
    {
        py::gil_scoped_release release;

        return camera_.StartCapture();
    }


    void Stop()
    {
        py::gil_scoped_release release;

        camera_.StopCapture();
    }


    void Close()
    {
        py::gil_scoped_release release;

        camera_.Close();
    }


    py::object GetLatestFrame()
    {
        RGBFrame frame;
        std::uint64_t sequence = 0;

        bool frameOK = false;

        /*
            GetLatestFrame copia el buffer C++.

            No necesita tener el GIL mientras realiza
            esa operación puramente nativa.
        */
        {
            py::gil_scoped_release release;

            frameOK = camera_.GetLatestFrame(
                frame,
                &sequence
            );
        }

        if (!frameOK)
        {
            return py::none();
        }

        const LONGLONG timestamp =
            frame.timestamp;

        py::array_t<std::uint8_t> numpyFrame =
            RGBFrameToNumpy(
                std::move(frame)
            );

        return py::make_tuple(
            std::move(numpyFrame),
            sequence,
            timestamp
        );
    }


    py::object WaitForNextFrame(
        std::uint64_t lastSequence,
        int timeoutMilliseconds = 1000
    )
    {
        RGBFrame frame;

        std::uint64_t outputSequence = 0;

        bool frameOK = false;

        /*
            Esta llamada puede esperar hasta que llegue
            un frame nuevo.

            Es especialmente importante liberar el GIL
            durante esa espera.
        */
        {
            py::gil_scoped_release release;

            frameOK = camera_.WaitForNextFrame(
                frame,
                lastSequence,
                outputSequence,
                timeoutMilliseconds
            );
        }

        if (!frameOK)
        {
            /*
                None significa:
                - timeout
                - captura detenida
                - no llegó un frame posterior
            */
            return py::none();
        }

        const LONGLONG timestamp =
            frame.timestamp;

        py::array_t<std::uint8_t> numpyFrame =
            RGBFrameToNumpy(
                std::move(frame)
            );

        return py::make_tuple(
            std::move(numpyFrame),
            outputSequence,
            timestamp
        );
    }


    bool IsOpen() const noexcept
    {
        return camera_.IsOpen();
    }


    bool IsCapturing() const noexcept
    {
        return camera_.IsCapturing();
    }


    std::uint32_t Width() const noexcept
    {
        return camera_.Width();
    }


    std::uint32_t Height() const noexcept
    {
        return camera_.Height();
    }


    std::uint64_t CapturedFrameCount()
        const noexcept
    {
        return camera_.CapturedFrameCount();
    }


private:
    void ShutdownRuntime() noexcept
    {
        /*
            Primero cerramos la cámara y el hilo.
        */
        camera_.Close();

        if (mediaFoundationStarted_)
        {
            MFShutdown();

            mediaFoundationStarted_ = false;
        }

        if (ownsCOM_)
        {
            CoUninitialize();

            ownsCOM_ = false;
        }
    }


private:
    Camera camera_;

    bool ownsCOM_ = false;

    bool mediaFoundationStarted_ = false;
};

PYBIND11_MODULE(hskcamera, module)
{
    module.doc() =
        "Captura de webcam de baja latencia "
        "con Media Foundation";

    py::class_<PythonCamera>(
        module,
        "Camera"
    )
        .def(
            py::init<>()
        )

        .def(
            "open",
            &PythonCamera::Open,
            py::arg("camera_index"),
            py::arg("format_index"),
            py::arg("exposure") = -6,
            py::arg("list_formats") = false
        )

        .def(
            "start",
            &PythonCamera::Start
        )

        .def(
            "stop",
            &PythonCamera::Stop
        )

        .def(
            "close",
            &PythonCamera::Close
        )

        .def(
            "get_latest_frame",
            &PythonCamera::GetLatestFrame
        )

        .def(
            "wait_for_next_frame",
            &PythonCamera::WaitForNextFrame,
            py::arg("last_sequence"),
            py::arg("timeout_ms") = 1000
        )

        .def_property_readonly(
            "is_open",
            &PythonCamera::IsOpen
        )

        .def_property_readonly(
            "is_capturing",
            &PythonCamera::IsCapturing
        )

        .def_property_readonly(
            "width",
            &PythonCamera::Width
        )

        .def_property_readonly(
            "height",
            &PythonCamera::Height
        )

        .def_property_readonly(
            "captured_frame_count",
            &PythonCamera::CapturedFrameCount
        );
}