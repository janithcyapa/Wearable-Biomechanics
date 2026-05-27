const EXPECTED_DEVICE_COUNT = 10;
const FILENAME_PATTERN = /^device([1-9]|10)_imu_(\d+)\.txt$/i;
const SENSOR_PATTERN = /^([A-Za-z]+)(\d+)$/;
const INTEGER_PATTERN = /^\d+$/;
const NUMBER_PATTERN = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i;

const state = {
  result: null,
};

async function readFiles(fileList) {
  const files = Array.from(fileList || []);
  return Promise.all(
    files.map(async (file) => ({
      name: file.name,
      text: await file.text(),
    })),
  );
}

function combineDeviceFiles(files) {
  if (!Array.isArray(files) || files.length === 0) {
    throw new Error("Select the 10 device .txt files first.");
  }

  const parsedFiles = files.map(parseDeviceFile);
  const errors = validateFiles(parsedFiles);
  if (errors.length > 0) {
    const error = new Error(errors.join("\n"));
    error.details = parsedFiles;
    throw error;
  }

  const sortedFiles = parsedFiles.sort((a, b) => a.deviceNumber - b.deviceNumber);
  const timestamp = sortedFiles[0].fileTimestamp;
  const commonTimestamps = getCommonTimestamps(sortedFiles);

  if (commonTimestamps.length === 0) {
    const error = new Error("No shared Timestamp values were found across all files.");
    error.details = sortedFiles;
    throw error;
  }

  const header = ["Timestamp"];
  for (const file of sortedFiles) {
    header.push(...file.dataColumns.map((column) => toWideColumnName(file.deviceName, column)));
  }

  const rows = commonTimestamps.map((timestampValue) => {
    const row = [timestampValue];
    for (const file of sortedFiles) {
      row.push(...file.rowsByTimestamp.get(timestampValue));
    }
    return row;
  });

  return {
    csv: toCsv([header, ...rows]),
    fileName: `combined_log_wide_${timestamp}.csv`,
    timestamp,
    commonRowCount: rows.length,
    files: sortedFiles,
    warnings: getWarnings(sortedFiles, commonTimestamps.length),
  };
}

function parseDeviceFile(file) {
  const match = file.name.match(FILENAME_PATTERN);
  const lines = normalizeLines(file.text)
    .map((line) => line.trim())
    .filter((line) => !shouldIgnoreLine(line));

  if (lines.length === 0) {
    return {
      name: file.name,
      error: "File is empty.",
    };
  }

  const headers = splitCsvFields(lines[0]);
  const timestampIndex = headers.indexOf("Timestamp");
  const dataColumns = headers.filter((header) => header !== "Timestamp");
  const rowsByTimestamp = new Map();
  const duplicateTimestamps = [];

  if (timestampIndex === -1) {
    return {
      name: file.name,
      error: "Header must include Timestamp.",
    };
  }

  for (const [rowIndex, line] of lines.slice(1).entries()) {
    const values = splitCsvFields(line);
    const rowNumber = rowIndex + 1;

    if (values.length !== headers.length) {
      return {
        name: file.name,
        error: `Row ${rowNumber} has ${values.length} comma-separated values, but expected ${headers.length}: Timestamp plus ${dataColumns.length} sensor values.`,
      };
    }

    const timestampValue = values[timestampIndex];
    if (!INTEGER_PATTERN.test(timestampValue)) {
      return {
        name: file.name,
        error: `Row ${rowNumber} Timestamp must be an integer value, found "${timestampValue}".`,
      };
    }

    const badValue = findBadSensorValue(headers, values);
    if (badValue) {
      return {
        name: file.name,
        error: `Row ${rowNumber} ${badValue.header} must be a numeric float value, found "${badValue.value}".`,
      };
    }

    if (rowsByTimestamp.has(timestampValue)) {
      duplicateTimestamps.push(timestampValue);
      continue;
    }

    rowsByTimestamp.set(
      timestampValue,
      headers
        .map((header, index) => ({ header, value: values[index] }))
        .filter((item) => item.header !== "Timestamp")
        .map((item) => item.value),
    );
  }

  return {
    name: file.name,
    deviceNumber: match ? Number(match[1]) : null,
    deviceName: match ? `device${Number(match[1])}` : null,
    fileTimestamp: match ? match[2] : null,
    headers,
    dataColumns,
    rowCount: rowsByTimestamp.size,
    rowsByTimestamp,
    imuCount: countImus(dataColumns),
    duplicateTimestamps,
    error: null,
  };
}

function findBadSensorValue(headers, values) {
  for (let index = 0; index < headers.length; index += 1) {
    if (headers[index] === "Timestamp") {
      continue;
    }

    if (!NUMBER_PATTERN.test(values[index])) {
      return {
        header: headers[index],
        value: values[index],
      };
    }
  }

  return null;
}

function validateFiles(files) {
  const errors = [];
  const validFiles = files.filter((file) => !file.error);

  for (const file of files) {
    if (file.error) {
      errors.push(`${file.name}: ${file.error}`);
    }

    if (!FILENAME_PATTERN.test(file.name)) {
      errors.push(`${file.name}: expected filename like device1_imu_1758111095.txt.`);
    }
  }

  if (validFiles.length !== EXPECTED_DEVICE_COUNT) {
    errors.push(`Expected ${EXPECTED_DEVICE_COUNT} files, but selected ${validFiles.length}.`);
  }

  const timestamps = new Set(validFiles.map((file) => file.fileTimestamp));
  if (timestamps.size > 1) {
    errors.push(`All filenames must use the same timestamp. Found: ${Array.from(timestamps).join(", ")}.`);
  }

  const seenDevices = new Set();
  for (const file of validFiles) {
    if (seenDevices.has(file.deviceNumber)) {
      errors.push(`Duplicate file for device${file.deviceNumber}.`);
    }

    seenDevices.add(file.deviceNumber);

    if (file.headers[0] !== "Timestamp") {
      errors.push(`${file.name}: first column must be Timestamp.`);
    }

    const badColumns = file.dataColumns.filter((column) => !SENSOR_PATTERN.test(column));
    if (badColumns.length > 0) {
      errors.push(`${file.name}: unsupported columns ${badColumns.join(", ")}.`);
    }

    if (file.duplicateTimestamps.length > 0) {
      errors.push(`${file.name}: duplicate Timestamp values found.`);
    }
  }

  for (let device = 1; device <= EXPECTED_DEVICE_COUNT; device += 1) {
    if (!seenDevices.has(device)) {
      errors.push(`Missing device${device}.`);
    }
  }

  return errors;
}

function getCommonTimestamps(files) {
  let common = new Set(files[0].rowsByTimestamp.keys());
  for (const file of files.slice(1)) {
    common = new Set(Array.from(common).filter((timestamp) => file.rowsByTimestamp.has(timestamp)));
  }

  return Array.from(common).sort(compareTimestamps);
}

function compareTimestamps(a, b) {
  const aNumber = Number(a);
  const bNumber = Number(b);
  if (Number.isFinite(aNumber) && Number.isFinite(bNumber)) {
    return aNumber - bNumber;
  }

  return a.localeCompare(b);
}

function toWideColumnName(deviceName, column) {
  const match = column.match(SENSOR_PATTERN);
  if (!match) {
    return `${deviceName}_${column}`;
  }

  const sensor = match[1].toUpperCase();
  const imuNumber = match[2];
  return `${deviceName}_IMU${imuNumber}_${sensor}`;
}

function countImus(columns) {
  const imus = new Set();
  for (const column of columns) {
    const match = column.match(SENSOR_PATTERN);
    if (match) {
      imus.add(match[2]);
    }
  }

  return imus.size;
}

function getWarnings(files, commonRowCount) {
  const warnings = [];
  const rowCounts = new Set(files.map((file) => file.rowCount));
  if (rowCounts.size > 1) {
    warnings.push(`Files have different row counts, so the CSV uses ${commonRowCount} shared Timestamp rows.`);
  }

  return warnings;
}

function normalizeLines(text) {
  return String(text || "").replace(/^\uFEFF/, "").split(/\r?\n/);
}

function shouldIgnoreLine(line) {
  const text = String(line || "").trim();
  const upperText = text.toUpperCase();

  return (
    text.length === 0 ||
    upperText === "FILE_END" ||
    upperText === "---END---" ||
    upperText.startsWith("IMUS_STATUS:")
  );
}

function splitCsvFields(line) {
  const values = String(line)
    .split(",")
    .map((value) => value.trim());

  while (values.length > 0 && values[values.length - 1] === "") {
    values.pop();
  }

  return values;
}

function toCsv(rows) {
  return rows.map((row) => row.map(escapeCsvValue).join(",")).join("\n");
}

function escapeCsvValue(value) {
  const text = String(value ?? "");
  if (/[",\n\r]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }

  return text;
}

function setupUi() {
  const fileInput = document.getElementById("file-input");
  const dropZone = document.getElementById("drop-zone");
  const resetButton = document.getElementById("reset-button");
  const downloadButton = document.getElementById("download-button");

  fileInput.addEventListener("change", async () => {
    await handleFiles(fileInput.files);
  });

  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragging");
  });

  dropZone.addEventListener("drop", async (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
    await handleFiles(event.dataTransfer.files);
  });

  resetButton.addEventListener("click", () => {
    state.result = null;
    fileInput.value = "";
    renderEmpty();
  });

  downloadButton.addEventListener("click", () => {
    if (!state.result) {
      return;
    }

    downloadCsv(state.result.csv, state.result.fileName);
  });
}

async function handleFiles(fileList) {
  renderWorking(fileList.length);
  try {
    const files = await readFiles(fileList);
    state.result = combineDeviceFiles(files);
    renderResult(state.result);
  } catch (error) {
    state.result = null;
    renderError(error);
  }
}

function renderEmpty() {
  setText("file-count", `0 / ${EXPECTED_DEVICE_COUNT}`);
  setText("timestamp-value", "-");
  setText("row-count", "-");
  setDisabled("download-button", true);
  setDisabled("reset-button", true);
  document.getElementById("details").hidden = true;
  document.getElementById("message-list").replaceChildren();
  document.getElementById("file-table-body").replaceChildren();
}

function renderWorking(count) {
  setText("file-count", `${count} / ${EXPECTED_DEVICE_COUNT}`);
  setText("timestamp-value", "Checking");
  setText("row-count", "-");
  setDisabled("download-button", true);
  setDisabled("reset-button", false);
}

function renderResult(result) {
  setText("file-count", `${result.files.length} / ${EXPECTED_DEVICE_COUNT}`);
  setText("timestamp-value", result.timestamp);
  setText("row-count", String(result.commonRowCount));
  setDisabled("download-button", false);
  setDisabled("reset-button", false);
  renderDetails(result.files, [
    { type: "ok", text: `${result.fileName} is ready.` },
    ...result.warnings.map((text) => ({ type: "warning", text })),
  ]);
}

function renderError(error) {
  setText("timestamp-value", "-");
  setText("row-count", "-");
  setDisabled("download-button", true);
  setDisabled("reset-button", false);

  const files = Array.isArray(error.details) ? error.details.filter((file) => file.deviceName) : [];
  const messages = String(error.message)
    .split("\n")
    .filter(Boolean)
    .map((text) => ({ type: "error", text }));

  setText("file-count", `${files.length} / ${EXPECTED_DEVICE_COUNT}`);
  renderDetails(files, messages);
}

function renderDetails(files, messages) {
  const details = document.getElementById("details");
  const messageList = document.getElementById("message-list");
  const tableBody = document.getElementById("file-table-body");

  messageList.replaceChildren(
    ...messages.map((message) => {
      const element = document.createElement("div");
      element.className = `message ${message.type}`;
      element.textContent = message.text;
      return element;
    }),
  );

  tableBody.replaceChildren(
    ...files.map((file) => {
      const row = document.createElement("tr");
      row.append(
        td(file.name),
        td(file.deviceName || "-"),
        td(file.fileTimestamp || "-"),
        td(file.rowCount ?? "-"),
        td(file.imuCount ?? "-"),
      );
      return row;
    }),
  );

  details.hidden = false;
}

function setText(id, text) {
  document.getElementById(id).textContent = text;
}

function setDisabled(id, disabled) {
  document.getElementById(id).disabled = disabled;
}

function td(text) {
  const cell = document.createElement("td");
  cell.textContent = text;
  return cell;
}

function downloadCsv(csv, fileName) {
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

if (typeof document !== "undefined") {
  setupUi();
}

globalThis.CombinedFile = {
  combineDeviceFiles,
  readFiles,
};
