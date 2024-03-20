import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from accelerate import Accelerator
import os
import glob
import transformers
import logging

print("Répertoire de travail actuel :", os.getcwd())
#transformers.logging.set_verbosity_debug()
#logging.basicConfig(level=logging.DEBUG)

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels['input_ids'][idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels['input_ids'])

def train_model(tokenized_data_folder, model_name="EleutherAI/gpt-neo-2.7B", output_dir="transpiler/transpile_project/model"):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)

        print(f"Chemin du dossier des données tokenisées : {tokenized_data_folder}")
        print(f"Chemin absolu : {os.path.abspath(tokenized_data_folder)}")

        if os.path.isdir(tokenized_data_folder):
            input_files = glob.glob(os.path.join(tokenized_data_folder, '*_inputs.pt'))
            target_files = glob.glob(os.path.join(tokenized_data_folder, '*_targets.pt'))

            print(f"Fichiers d'entrée trouvés : {input_files}")
            print(f"Fichiers cible trouvés : {target_files}")

            if not input_files or not target_files:
                raise FileNotFoundError(f"Aucun fichier .pt trouvé dans {tokenized_data_folder}")

            combined_encodings = {'input_ids': [], 'attention_mask': []}
            combined_labels = {'input_ids': []}
            for input_file, target_file in zip(input_files, target_files):
                inputs = torch.load(input_file)
                targets = torch.load(target_file)
                combined_encodings['input_ids'].extend(inputs['input_ids'])
                combined_encodings['attention_mask'].extend(inputs['attention_mask'])
                combined_labels['input_ids'].extend(targets['input_ids'])

            train_dataset = CustomDataset(combined_encodings, combined_labels)

            # Vérification et lecture d'un fichier spécifique
            #print(f"Contenu du dossier {tokenized_data_folder}:")
            #for file in os.listdir(tokenized_data_folder):
                #print(file)

            #specific_file_path = os.path.join(tokenized_data_folder, 'test_inputs.pt')
            #if os.path.isfile(specific_file_path):
                #print(f"Le fichier {specific_file_path} existe.")
                #try:
                    #with open(specific_file_path, 'rb') as f:
                        #print(f"Contenu du fichier {specific_file_path} :")
                        #print(f.read())
                #except Exception as e:
                    #print(f"Erreur lors de la tentative de lecture du fichier {specific_file_path} : {e}")
            #else:
                #print(f"Le fichier {specific_file_path} n'existe pas.")
        else:
            raise FileNotFoundError(f"Le dossier spécifié {tokenized_data_folder} n'existe pas ou ne contient pas de fichiers .pt")
        
        logs_folder = 'transpiler/transpile_project/Logs'

        logs_folder = os.path.abspath(logs_folder)

        if not os.path.exists(logs_folder):
            os.makedirs(logs_folder)
            print(f"Le dossier {logs_folder} a été créé.")
        else :
            print(f"Le dossier {logs_folder} existe déjà.")

        if not os.path.isdir(logs_folder):
            raise ValueError(f"Le chemin {logs_folder} n'est pas un dossier.")

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            num_train_epochs=3,
            logging_dir=logs_folder,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            gradient_checkpointing=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Initialiser l'Accelerator
        accelerator = Accelerator()
        print(f"Entraînement sur: {accelerator.device}")
        trainer = accelerator.prepare(trainer)

        trainer.train()

        model.save_pretrained(output_dir)

    except Exception as e:
        print(f"Une exception a été rencontrée : {e}")

if __name__ == "__main__":
    try:
        tokenized_data_folder = 'transpiler/transpile_project/tokenized_data'
        train_model(tokenized_data_folder)
    except Exception as e:
        print(f"Erreur lors de l'exécution du script : {e}")
