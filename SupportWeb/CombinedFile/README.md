# Combined File

Browser tool for combining 10 IMU `.txt` files into one wide CSV.

## Use

Open `index.html` in a browser, upload `device1_imu_<timestamp>.txt` through `device10_imu_<timestamp>.txt`, then click **Download CSV**.

The generated filename is:

```text
combined_log_wide_<timestamp>.csv
```

The app checks that all 10 files are present, the device numbers are unique, and every filename has the same timestamp. Rows are joined by the shared `Timestamp` values across all device files. Output columns are ordered numerically from `device1` through `device10`.

Every data row must include the `Timestamp` value as the first comma-separated field, followed by the sensor values listed in the header. `Timestamp` must be an integer, and sensor values must be numeric float-compatible values. For example, a 2-IMU file has 13 fields per row: `Timestamp` plus 12 sensor values. Trailing commas at the end of rows are ignored. Footer lines such as `---END---` and `IMUs_status: ...` are skipped.

## Test

From this folder:

```powershell
npm test
```
