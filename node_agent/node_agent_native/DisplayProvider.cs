using System;
using System.Collections.Generic;
using System.Management;
using System.Runtime.InteropServices;

namespace HveacNodeAgentNative
{
    public class DisplayProvider
    {
        [DllImport("user32.dll", CharSet = CharSet.Ansi)]
        private static extern bool EnumDisplayDevices(string? lpDevice, uint iDevNum, ref DISPLAY_DEVICE lpDisplayDevice, uint dwFlags);

        [DllImport("user32.dll")]
        private static extern bool EnumDisplaySettings(string deviceName, int modeNum, ref DEVMODE devMode);

        private const int ENUM_CURRENT_SETTINGS = -1;
        private const int DISPLAY_DEVICE_ATTACHED_TO_DESKTOP = 0x00000001;

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
        private struct DISPLAY_DEVICE
        {
            [MarshalAs(UnmanagedType.U4)]
            public int cb;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
            public string DeviceName;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
            public string DeviceString;
            [MarshalAs(UnmanagedType.U4)]
            public int StateFlags;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
            public string DeviceID;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
            public string DeviceKey;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct DEVMODE
        {
            private const int CCHDEVICENAME = 32;
            private const int CCHFORMNAME = 32;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = CCHDEVICENAME)]
            public string dmDeviceName;
            public short dmSpecVersion;
            public short dmDriverVersion;
            public short dmSize;
            public short dmDriverExtra;
            public int dmFields;
            public int dmPositionX;
            public int dmPositionY;
            public int dmDisplayOrientation;
            public int dmDisplayFixedOutput;
            public short dmColor;
            public short dmDuplex;
            public short dmYResolution;
            public short dmTTOption;
            public short dmCollate;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = CCHFORMNAME)]
            public string dmFormName;
            public short dmLogPixels;
            public int dmBitsPerPel;
            public int dmPelsWidth;
            public int dmPelsHeight;
            public int dmDisplayFlags;
            public int dmDisplayFrequency;
        }

        private DateTime _lastBrightnessCheck = DateTime.MinValue;
        private double? _cachedBrightness = null;

        public void PopulateDisplayData(DisplayData displayData)
        {
            var displays = new List<DisplayDeviceData>();
            var d = new DISPLAY_DEVICE();
            d.cb = Marshal.SizeOf(d);

            try
            {
                for (uint id = 0; EnumDisplayDevices(null, id, ref d, 0); id++)
                {
                    if ((d.StateFlags & DISPLAY_DEVICE_ATTACHED_TO_DESKTOP) != 0)
                    {
                        var dm = new DEVMODE();
                        dm.dmSize = (short)Marshal.SizeOf(typeof(DEVMODE));

                        if (EnumDisplaySettings(d.DeviceName, ENUM_CURRENT_SETTINGS, ref dm))
                        {
                            // Also query monitor name for this adapter device
                            string monitorName = d.DeviceString;
                            var monDevice = new DISPLAY_DEVICE();
                            monDevice.cb = Marshal.SizeOf(monDevice);
                            if (EnumDisplayDevices(d.DeviceName, 0, ref monDevice, 0) && !string.IsNullOrWhiteSpace(monDevice.DeviceString))
                            {
                                monitorName = monDevice.DeviceString.Trim();
                            }

                            displays.Add(new DisplayDeviceData
                            {
                                DisplayId = d.DeviceName,
                                DisplayName = monitorName,
                                ResolutionWidth = dm.dmPelsWidth,
                                ResolutionHeight = dm.dmPelsHeight,
                                Resolution = $"{dm.dmPelsWidth}x{dm.dmPelsHeight}",
                                RefreshRateHz = dm.dmDisplayFrequency > 0 ? dm.dmDisplayFrequency : null,
                                BrightnessPercent = null
                            });
                        }
                    }
                    d.cb = Marshal.SizeOf(d);
                }
            }
            catch { }

            // Brightness caching (every 10s to keep cycle fast)
            if ((DateTime.UtcNow - _lastBrightnessCheck).TotalSeconds > 10)
            {
                _cachedBrightness = QueryWmiBrightness();
                _lastBrightnessCheck = DateTime.UtcNow;
            }

            // Assign brightness if primary display exists and brightness was detected
            if (displays.Count > 0 && _cachedBrightness.HasValue)
            {
                displays[0].BrightnessPercent = _cachedBrightness.Value;
            }

            displayData.Displays = displays;
            displayData.DisplayCount = displays.Count;
        }

        private static double? QueryWmiBrightness()
        {
            try
            {
                using var searcher = new ManagementObjectSearcher("root\\WMI", "SELECT CurrentBrightness FROM WmiMonitorBrightness");
                using var instances = searcher.Get();
                foreach (ManagementObject instance in instances)
                {
                    if (instance["CurrentBrightness"] != null)
                    {
                        return Convert.ToDouble(instance["CurrentBrightness"]);
                    }
                }
            }
            catch
            {
                // Not supported on desktops or without supported monitor DDC/CI driver
            }
            return null;
        }
    }
}
