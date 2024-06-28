import pickle

def read_history(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def save_history_to_txt(history_data, output_file):
    with open(output_file, 'w') as file:
        for key, values in history_data.items():
            file.write(f"{key}: {values}\n")

if __name__ == "__main__":
    history_file = 'ml-20m_mult_64.his'
    output_file = 'ml-20m_mult_64.txt'
    
    history_data = read_history(history_file)
    save_history_to_txt(history_data, output_file)

    print(f"History data has been saved to {output_file}")
