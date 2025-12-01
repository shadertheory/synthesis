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

template <size_t N, typename mode> struct tensor
{
	mode inner[N];

	mode& operator[](size_t i)
	{
		return inner[i];
	}
	const mode& operator[](size_t i) const
	{
		return inner[i];
	}
};

template <typename scalar> struct tensor<0, scalar>
{
	scalar value;

	operator scalar&()
	{
		return value;
	}
	operator const scalar&() const
	{
		return value;
	}
};

template <typename of>
concept fp = floating_point<of>;
template <typename of>
concept bits = integral<of>;
template <typename of>
concept number = fp<of> || bits<of>;

template <number repr> using scalar = tensor<0, repr>;

template <usize dimensions, number repr = f32> using vector = tensor<dimensions, scalar<repr>>;

template <usize rows, usize cols = rows, number repr = f32> using matrix = tensor<rows, vector<cols, repr>>;

template <bits data> usize count_leading_zeros(data* value);

extern "C"
{
	// MEMORY
	void* allocate_aligned(usize size, usize alignment)
	{
	}
	void* allocate_any(usize size)
	{
		constexpr usize ALIGN = 16;
		usize align = min(ALIGN, count_leading_zeros(&size));
		return allocate_aligned(size, align);
	}

	void* reallocate(void* old, usize new_size)
	{
	}

	void deallocate(void* ptr)
	{
	}

	// ASYNC
	struct raw_waker;

	typedef struct raw_waker_virtual_table
	{
		void (*clone)(const void* data, raw_waker* out_waker);
		void (*wake)(const void* data);
		void (*wake_by_ref)(const void* data);
		void (*drop)(const void* data);
	} waker_virtual_table;

	struct raw_waker
	{
		const raw_waker_virtual_table* vtable;
		const void* data;
	};

	typedef enum
	{
		POLL_RESULT_READY,
		POLL_RESULT_PENDING
	} poll_result;

	struct task_virtual_table
	{
		poll_result (*poll)(void* task_data, raw_waker* waker);
		void (*drop)(void* task_data);
	};

	struct task
	{
		task_virtual_table* vtable;
		void* data;
	};

	struct handle
	{
		usize id;
	};

	handle* task_spawn(task task);

	// INPUT
	enum cursor_mode
	{
		CURSOR_MODE_VISIBLE,
		CURSOR_MODE_HIDDEN,
		CURSOR_MODE_CAPTURED
	};

	enum mouse_button
	{
		MOUSE_BUTTON_LEFT,
		MOUSE_BUTTON_RIGHT,
		MOUSE_BUTTON_MIDDLE
	};

	enum axis
	{
		AXIS_X,
		AXIS_Y,
		AXIS_Z,
	};

	enum gamepad_button
	{
		GAMEPAD_BUTTON_DOWN,
		GAMEPAD_BUTTON_RIGHT,
		GAMEPAD_BUTTON_UP,
		GAMEPAD_BUTTON_LEFT,
		GAMEPAD_BUTTON_RIGHT_STICK,
		GAMEPAD_BUTTON_LEFT_STICK,
		GAMEPAD_BUTTON_RIGHT_BUMPER,
		GAMEPAD_BUTTON_LEFT_BUMPER
	};

	enum gamepad_analog
	{
		GAMEPAD_ANALOG_LEFT_STICK,
		GAMEPAD_ANALOG_RIGHT_STICK,
		GAMEPAD_ANALOG_LEFT_TRIGGER,
		GAMEPAD_ANALOG_RIGHT_TRIGGER
	};

	using scan_code = u16;

	enum button_state
	{
		DOWN,
		UP,
		PRESS,
		TAP
	};

	bool input_scan(scan_code key, button_state state);

	scalar<f32> input_mouse_delta(axis which);
	void input_mouse_delta_reset();
	void input_mouse_cursor(cursor_mode mode);
	void input_mouse_button(mouse_button button, button_state state);

	bool input_gamepad_connected(usize index);
	float input_gamepad_analog(usize index, gamepad_analog analog, axis which);
	bool input_gamepad_button(usize index, gamepad_button button, button_state state);

	usize time_current();
	usize time_tick_rate();
	usize time_one_second();

	void* window_handle();
	vector<2, i32> window_size();
	scalar<f32> window_scale();
	void window_title(const char* title);
	void window_fullscreen(bool engage);

	struct Descriptor;
	using handle = Descriptor*;

	enum access_mode
	{
		FILE_READ,
		FILE_WRITE
	};

	handle file_open(const char* path, access_mode mode)
	{
	}

	enum seek_origin
	{
		SEEK_ORIGIN_START,
		SEEK_ORIGIN_END,
	};

	usize file_read(handle file, void* buffer, usize size);
	usize file_write(handle file, const void* buffer, usize size);
	void file_seek(handle file, usize offset, seek_origin origin);
	usize file_size(handle file);
	void file_close(handle file);
	bool file_exists(handle file);

	/**
	*
	- INPUT (POLLING)


	- FILE SYSTEM (VFS)
	 - Filehandle file_open(const char* path, Fileaccess_mode mode)
	 - size_t     file_read(Filehandle handle, void* buffer, size_t bytes)
	 - size_t     file_write(Filehandle handle, const void* buffer, size_t bytes)
	 - void       file_seek(Filehandle handle, size_t offset, SeekOrigin origin)
	 - size_t     file_tell(Filehandle handle)
	 - void       file_close(Filehandle handle)
	 - bool       file_exists(const char* path)

	 // Path utilities relying on OS separators
	 - void       path_get_executable_dir(char* buffer, size_t max_len)
	 - void       path_get_user_data_dir(char* buffer, size_t max_len)

	- AUDIO (BASIC)
	 // Requesting the platform to mix a raw buffer
	 - void     audio_push_buffer(const int16_t* samples, size_t frame_count, int
	channels, int sample_rate)
	 - void     audio_set_master_volume(float volume)

	- DIAGNOSTICS
	 - void     debug_print(const char* message)
	 - void     debug_break()               // Intrinsic breakpoint
	 - void     system_show_alert(const char* title, const char* message) // Native
	message box
	*/
}

// MSVC specific
template <integral bits> usize count_leading_zeros(bits* data)
{
	return BitScanForward64(data, !(bits)0);
}
