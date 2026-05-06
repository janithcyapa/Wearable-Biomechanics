import os
import json
import io
import pandas as pd
from datetime import datetime
from google.colab import files
from IPython.display import display

# Default base directory
DEFAULT_BASE_DIR = '/content/drive/MyDrive/Datasets/Wearable-Biomechanics'

class SubjectManager:
    def __init__(self, base_dir=DEFAULT_BASE_DIR):
        self.base_dir = base_dir
        self.file_path = f'{self.base_dir}/subjects.json'

        os.makedirs(self.base_dir, exist_ok=True)

        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump({}, f)
            print(f"Created a new subjects.json file in {self.base_dir}.")
        else:
            print(f"Connected to existing subjects.json in {self.base_dir}.")

    def _load_data(self):
        with open(self.file_path, 'r') as f:
            return json.load(f)

    def _save_data(self, data):
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def add_subject(self, subject_id, name, weight, height, max_lift, description):
        data = self._load_data()
        if subject_id in data:
            print(f"⚠️ Error: Subject ID '{subject_id}' already exists!")
            return

        data[subject_id] = {
            "name": name,
            "weight_kg": weight,
            "height_cm": height,
            "max_lift_kg": max_lift,
            "description": description
        }
        self._save_data(data)
        print(f"✅ Successfully added {name} ({subject_id}).")

    def list_subjects(self, subject_id=None):
        data = self._load_data()
        if subject_id:
            return data.get(subject_id, "⚠️ Subject not found.")

        print(f"\n--- Database contains {len(data)} subjects ---")
        return data

    def update_subject(self, subject_id, **kwargs):
        data = self._load_data()
        if subject_id not in data:
            print(f"⚠️ Error: Subject ID '{subject_id}' not found.")
            return

        for key, value in kwargs.items():
            data[subject_id][key] = value

        self._save_data(data)
        print(f"✅ Successfully updated {subject_id}.")

    def delete_subject(self, subject_id):
        data = self._load_data()
        if subject_id in data:
            del data[subject_id]
            self._save_data(data)
            print(f"🗑️ Successfully deleted {subject_id}.")
        else:
            print(f"⚠️ Error: Subject ID '{subject_id}' not found.")


class ExperimentManager:
    def __init__(self, base_dir=DEFAULT_BASE_DIR):
        self.base_dir = base_dir
        self.index_dir = f'{self.base_dir}/db_index'
        self.raw_dir = f'{self.base_dir}/raw_data'
        self.file_path = f'{self.index_dir}/experiments.json'

        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)

        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump({}, f)
            print("Created a new experiments.json file.")
        else:
            print("Connected to experiments database.")

    def _load_data(self):
        with open(self.file_path, 'r') as f:
            return json.load(f)

    def _save_data(self, data):
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def interactive_upload(self, experiment_id, subject_id, date, time, activity_type, description):
        data = self._load_data()

        if experiment_id in data:
            print(f"⚠️ Error: Experiment ID '{experiment_id}' already exists.")
            return

        print(f"📂 Please select the raw CSV file for {experiment_id}...")
        uploaded = files.upload()

        if not uploaded:
            print("⚠️ Upload cancelled.")
            return

        original_filename = list(uploaded.keys())[0]

        try:
            raw_dataframe = pd.read_csv(io.BytesIO(uploaded[original_filename]))
        except Exception as e:
            print(f"⚠️ Error reading CSV: {e}")
            return

        clean_date = date.replace("-", "").replace("/", "")
        new_filename = f"{subject_id}_{experiment_id}_{clean_date}.csv"

        csv_rel_path = f"raw_data/{new_filename}"
        csv_full_path = f"{self.raw_dir}/{new_filename}"

        raw_dataframe.to_csv(csv_full_path, index=False)

        data[experiment_id] = {
            "subject_id": subject_id,
            "date": date,
            "time": time,
            "activity_type": activity_type,
            "description": description,
            "raw_csv_path": csv_rel_path,
            "processed_files": {}
        }

        self._save_data(data)
        del uploaded[original_filename]
        print(f"✅ Success! Data uploaded, renamed to '{new_filename}', and logged.")

    def list_experiments(self):
        data = self._load_data()
        if not data:
            print("Database is empty.")
            return

        table_rows = []
        for exp_id, info in data.items():
            proc_files = info.get('processed_files', {})
            if proc_files:
                proc_str = " | ".join([f"{tag}: {os.path.basename(path)}" for tag, path in proc_files.items()])
            else:
                proc_str = "❌ None yet"

            table_rows.append({
                'Experiment ID': exp_id,
                'Subject': info.get('subject_id', 'N/A'),
                'Date': info.get('date', 'N/A'),
                'Activity': info.get('activity_type', 'N/A'),
                'Raw File': info.get('raw_csv_path', 'N/A'),
                'Processed Files': proc_str
            })

        df = pd.DataFrame(table_rows)
        print(f"\n📊 Database contains {len(df)} experiments:")
        display(df)

    def delete_experiment(self, experiment_id):
        data = self._load_data()
        if experiment_id not in data:
            print(f"⚠️ Error: '{experiment_id}' not found.")
            return

        exp_info = data[experiment_id]
        print(f"🗑️ Initiating deletion for {experiment_id}...")

        raw_path = f"{self.base_dir}/{exp_info['raw_csv_path']}"
        if os.path.exists(raw_path):
            os.remove(raw_path)
            print(f"  - Deleted raw file: {exp_info['raw_csv_path']}")

        for tag, processed_path in exp_info['processed_files'].items():
            full_proc_path = f"{self.base_dir}/{processed_path}"
            if os.path.exists(full_proc_path):
                os.remove(full_proc_path)
                print(f"  - Deleted processed [{tag}] file.")

        del data[experiment_id]
        self._save_data(data)
        print(f"✅ Experiment '{experiment_id}' completely removed.")

    def add_processed_data(self, experiment_id, tag, processed_df):
        data = self._load_data()

        if experiment_id not in data:
            print(f"⚠️ Error: Experiment '{experiment_id}' not found.")
            return

        save_dir = f"{self.base_dir}/processed_data/{tag}"
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{experiment_id}_{tag}.parquet"
        full_path = f"{save_dir}/{filename}"
        rel_path = f"processed_data/{tag}/{filename}"

        processed_df.to_parquet(full_path, engine='pyarrow', index=False)

        data[experiment_id]['processed_files'][tag] = rel_path
        self._save_data(data)
        print(f"✅ Processed data '{tag}' saved and linked to {experiment_id}!")