//
// Main.cpp
//

#include "pch.h"

#ifdef __clang__
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#pragma clang diagnostic ignored "-Wswitch-enum"
#endif

#pragma warning(disable : 4061)

int entry();

// Entry point
#ifdef _MSC_VER
int WINAPI wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine,
					_In_ int nCmdShow)
{
	return entry();
}
#else
int main()
{
	return entry();
}
#endif

int entry()
{
}



// MSVC specific
template <integral bits> usize count_leading_zeros(bits* data)
{
	return BitScanForward64(data, !(bits)0);
}
