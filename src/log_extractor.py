import re
from datetime import datetime

def parse_log_file(log_file_path):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    table = {
       'epoch_time': [],
       'val_loss': [],
       'val_accuracy': [],
       'loss': [],
       'accuracy': []
    }

    for line in lines:

        # Extracting specific values
        match = re.search(r'Epoch training time: (.+)', line)
        if match:
            table['epoch_time'].append((match.group(1)))
            continue

        match = re.search(r'Validation Loss: (.+)', line)
        if match:
            table['val_loss'].append((match.group(1)))
            continue

        match = re.search(r'Validation Accuracy: (.+)%', line)
        if match:
            table['val_accuracy'].append((match.group(1)))
            continue

        match = re.search(r'Loss: (.+)', line)
        if match:
            table['loss'].append((match.group(1)))
            continue

        match = re.search(r'Accuracy: (.+)%', line)
        if match:
            table['accuracy'].append((match.group(1)))
            continue

    return table

if __name__ == "__main__":
    log_file_paths = [
        "../experiment1/reference.log",
        "../experiment1/split/alice1.log",
        "../experiment1/split/alice2.log",
        "../experiment2/a/reference.log",
        "../experiment2/a/split/alice1.log",
        "../experiment2/a/split/alice2.log",
        "../experiment2/b/reference.log",
        "../experiment2/b/split/alice1.log",
        "../experiment2/b/split/alice2.log",
        "../experiment3/reference_augment.log",
        "../experiment3/split/alice1.log",
        "../experiment3/split/alice2.log"
    ]
    csv_log_file_paths = [
        "../csv_logs/experiment1_ref.log",
        "../csv_logs/experiment1_alice1.log",
        "../csv_logs/experiment1_alice2.log",
        "../csv_logs/experiment2a_ref.log",
        "../csv_logs/experiment2a_alice1.log",
        "../csv_logs/experiment2a_alice2.log",
        "../csv_logs/experiment2b_ref.log",
        "../csv_logs/experiment2b_alice1.log",
        "../csv_logs/experiment2b_alice2.log",
        "../csv_logs/experiment3_ref.log",
        "../csv_logs/experiment3_alice1.log",
        "../csv_logs/experiment3_alice2.log",
    ]

    for log_file_path in log_file_paths:
        extracted_data = parse_log_file(log_file_path)
        csv_content = ""
        with open("../csv_logs/experiments.csv", 'a') as file:
            file.write(log_file_path.split("/")[-1] + "\n")
            for entry in extracted_data:
                file.write(entry + ":," + ",".join(extracted_data[entry]) + "\n")

