#include "utils.h"

#include <locale>
#include <codecvt>

// STB library implementation
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

void utils::validate(HRESULT hr, LPCWSTR msg)
{
	if (FAILED(hr))
	{
		MessageBox(NULL, msg, L"Error", MB_OK);
		PostQuitMessage(EXIT_FAILURE);
	}
}

std::wstring utils::extractExtension(std::wstring filePath) {
	auto lastDot = wcsrchr(filePath.c_str(), '.');
	if (!lastDot) return L"";
	return std::wstring(lastDot + 1);
}

std::wstring utils::getPath(std::wstring filePath) {

	auto lastSlash = wcsrchr(filePath.c_str(), '\\');
	auto lastForwardSlash = wcsrchr(filePath.c_str(), '/');

	if (lastForwardSlash && (!lastSlash || lastSlash < lastForwardSlash))
		lastSlash = lastForwardSlash;

	if (!lastSlash) return L".\\";

	return std::wstring(filePath.c_str(), lastSlash + 1);
}

std::wstring utils::getExePath() {

	TCHAR exePath[MAX_PATH];
	if (GetModuleFileName(NULL, exePath, MAX_PATH)) {
		return getPath(exePath);
	}

	return L".\\";
}

std::string utils::wstringToString(const std::wstring& wstr)
{
	static std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
	return converter.to_bytes(wstr);
}

std::wstring utils::stringToWstring(const std::string& wstr)
{
	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
	return converter.from_bytes(wstr);
}

float utils::getDpiScale(HWND window) {

	unsigned int dpi = GetDpiForWindow(window);

	const float defaultDpi = 96.0f; //< Default monitor DPI of the yesteryear
	float dpiScale = dpi / defaultDpi;

	return dpiScale;
}

uint32_t utils::divRoundUp(uint32_t x, uint32_t div)
{
	return (x + div - 1) / div;
}