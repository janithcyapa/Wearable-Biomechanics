import { readFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import vm from "node:vm";

const root = process.cwd();
const scriptPath = join(root, "script.js");
const sampleDir = join(
  root,
  "..",
  "Gait_and_Posture",
  "Sample_Data",
  "0,0_Walking",
  "1757660235_12_09_2025_Flat_0,0_60Steps_JRDC_Trial01[Left ankle issue]",
);
const expectedPath = join(sampleDir, "combined_log_wide_1757660235.csv");

const names = (await readdir(sampleDir))
  .filter((name) => /^device\d+_imu_1757660235\.txt$/i.test(name))
  .sort();

const files = await Promise.all(
  names.map(async (name) => ({
    name,
    text: await readFile(join(sampleDir, name), "utf8"),
  })),
);

const context = { globalThis: {} };
vm.createContext(context);
vm.runInContext(await readFile(scriptPath, "utf8"), context);

const { combineDeviceFiles } = context.globalThis.CombinedFile;
const result = combineDeviceFiles(files);
const expected = await readFile(expectedPath, "utf8");
const expectedDataRows = normalize(expected).split("\n").length - 1;
const generatedLines = normalize(result.csv).split("\n");
const generatedColumns = generatedLines[0].split(",");

if (result.fileName !== "combined_log_wide_1757660235.csv") {
  throw new Error(`Unexpected file name: ${result.fileName}`);
}

if (result.commonRowCount !== expectedDataRows) {
  throw new Error(`Unexpected shared row count: ${result.commonRowCount}`);
}

if (generatedLines.length - 1 !== expectedDataRows) {
  throw new Error("Generated CSV row count does not match combined_log_wide_1757660235.csv.");
}

if (generatedColumns.length !== normalize(expected).split("\n")[0].split(",").length) {
  throw new Error("Generated CSV column count does not match combined_log_wide_1757660235.csv.");
}

const deviceIndexes = Array.from({ length: 10 }, (_, index) =>
  generatedColumns.indexOf(`device${index + 1}_IMU1_AX`),
);

if (deviceIndexes.some((index) => index === -1)) {
  throw new Error("Generated CSV is missing one or more device columns.");
}

if (!deviceIndexes.every((index, arrayIndex) => arrayIndex === 0 || index > deviceIndexes[arrayIndex - 1])) {
  throw new Error("Device columns are not in numeric order from device1 to device10.");
}

const timestampRowsWithTrailingComma = Array.from({ length: 10 }, (_, index) => {
  const device = index + 1;
  return {
    name: `device${device}_imu_123.txt`,
    text: "Timestamp,AX1,AY1\n1,10,20,\n2,11,21,\n\n---END---\n IMUs_status: SPI1=OK, SPI2=OK\n",
  };
});
const trailingCommaResult = combineDeviceFiles(timestampRowsWithTrailingComma);

if (!trailingCommaResult.csv.startsWith("Timestamp,device1_IMU1_AX,device1_IMU1_AY,device2_IMU1_AX")) {
  throw new Error("Rows are not combined in numeric device order.");
}

if (!trailingCommaResult.csv.includes("\n1,10,20,10,20")) {
  throw new Error("Timestamp rows with trailing commas should keep the uploaded Timestamp values.");
}

const badTimestampFiles = timestampRowsWithTrailingComma.map((file, index) => ({
  ...file,
  text: index === 0 ? "Timestamp,AX1,AY1\n1.5,10,20\n" : file.text,
}));

assertThrows(
  () => combineDeviceFiles(badTimestampFiles),
  "Timestamp must be an integer value",
);

const badSensorFiles = timestampRowsWithTrailingComma.map((file, index) => ({
  ...file,
  text: index === 0 ? "Timestamp,AX1,AY1\n1,not-a-number,20\n" : file.text,
}));

assertThrows(
  () => combineDeviceFiles(badSensorFiles),
  "AX1 must be a numeric float value",
);

console.log(`OK ${result.fileName}: numeric device order and timestamp rows supported.`);

function normalize(text) {
  return text.trim().replace(/\r\n/g, "\n");
}

function assertThrows(fn, expectedMessage) {
  try {
    fn();
  } catch (error) {
    if (String(error.message).includes(expectedMessage)) {
      return;
    }

    throw new Error(`Expected error containing "${expectedMessage}", found "${error.message}".`);
  }

  throw new Error(`Expected error containing "${expectedMessage}".`);
}
