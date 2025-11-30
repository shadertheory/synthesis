//
// pch.h
// Header for standard system include files.
//

#pragma once

#include <winsdkver.h>
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include <sdkddkver.h>

// Use the C++ standard templated min/max
#define NOMINMAX

// DirectX apps don't need GDI
#define NODRAWTEXT
#define NOGDI
#define NOBITMAP

// Include <mcx.h> if you need this
#define NOMCX

// Include <winsvc.h> if you need this
#define NOSERVICE

// WinHelp is deprecated
#define NOHELP

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <wrl/client.h>
#include <wrl/event.h>

#include <directx/dxgiformat.h>
#include <directx/d3d12.h>
#include <directx/d3dx12.h>
#include <dxguids/dxguids.h>

#include <dxgi1_6.h>

#include <DirectXMath.h>
#include <DirectXColors.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <exception>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <tuple>

#ifdef _DEBUG
#include <dxgidebug.h>
#endif

// WinPixEvent Runtime
#include <pix3.h>

// Game Runtime
#include <XGameRuntime.h>

#ifdef __M_X64
// GameInput
#include <gameinput.h>

// XSAPI
#include <xsapi-c/services_c.h>

#include <httpClient/httpClient.h>

#include <XCurl.h>

// If using Xbox GameChat, uncomment this line:
//#include <GameChat2.h>

// If using Azure PlayFab Services, uncommment these:
//#include <playfab/core/PFErrors.h>
//#include <playfab/services/PFServices.h>
#endif

// If using the DirectX Shader Compiler API, uncomment this line:
//#include <directx-dxc/dxcapi.h>

// If using DirectStorage, uncomment this line:
//#include <dstorage.h>

// If using the DirectX Tool Kit for DX12, uncomment this line:
//#include <directxtk12/GraphicsMemory.h>

namespace DX
{
    // Helper class for COM exceptions
    class com_exception : public std::exception
    {
    public:
        com_exception(HRESULT hr) noexcept : result(hr) {}

        const char* what() const noexcept override
        {
            static char s_str[64] = {};
            sprintf_s(s_str, "Failure with HRESULT of %08X", static_cast<unsigned int>(result));
            return s_str;
        }

    private:
        HRESULT result;
    };

    // Helper utility converts D3D API failures into exceptions.
    inline void ThrowIfFailed(HRESULT hr)
    {
        if (FAILED(hr))
        {
            throw com_exception(hr);
        }
    }
}
