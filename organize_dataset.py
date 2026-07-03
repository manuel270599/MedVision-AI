import os
import shutil
import random
from sklearn.model_selection import train_test_split

# ========== CONFIGURACIÓN ==========
# Rutas de origen (AJUSTA ESTAS RUTAS SEGÚN TU DESCARGA)
SOURCE_PNEUMONIA = r"C:\Users\rockm\Downloads\neumonia\chest_xray\train"  # Carpeta con NORMAL/ y PNEUMONIA/
SOURCE_TB = r"C:\Users\rockm\Downloads\archive\TB_Chest_Radiography_Database"  # Carpeta con Normal/ y Tuberculosis/

# Ruta de destino (donde se creará la carpeta 'dataset')
DEST_DIR = "dataset"  # Se creará en el directorio actual

# Proporción de validación (20% para validación)
VAL_SIZE = 0.2
RANDOM_SEED = 42

# ========== FUNCIONES ==========
def copy_files(src_dir, dest_dir, class_name, val_size=0.2):
    """
    Copia archivos de src_dir (que contiene archivos de una clase) a dest_dir/class_name/ 
    dividiendo en train y val según val_size.
    """
    # Crear carpetas de destino
    train_dir = os.path.join(dest_dir, "train", class_name)
    val_dir = os.path.join(dest_dir, "val", class_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Obtener lista de archivos de imagen
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print(f" No se encontraron imágenes en {src_dir}")
        return

    # Dividir en train y val
    train_files, val_files = train_test_split(files, test_size=val_size, random_state=RANDOM_SEED)

    # Copiar archivos
    for f in train_files:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(val_dir, f))

    print(f" {class_name}: {len(train_files)} train, {len(val_files)} val")

# ========== EJECUCIÓN ==========
if __name__ == "__main__":
    print(" Organizando datasets...")

    # 1. Neumonía (desde carpeta PNEUMONIA)
    copy_files(
        src_dir=os.path.join(SOURCE_PNEUMONIA, "PNEUMONIA"),
        dest_dir=DEST_DIR,
        class_name="Neumonia",
        val_size=VAL_SIZE
    )

    # 2. Normal (desde carpeta NORMAL del dataset de neumonía)
    copy_files(
        src_dir=os.path.join(SOURCE_PNEUMONIA, "NORMAL"),
        dest_dir=DEST_DIR,
        class_name="Normal",
        val_size=VAL_SIZE
    )

    # 3. Tuberculosis (desde carpeta Tuberculosis del dataset TB)
    copy_files(
        src_dir=os.path.join(SOURCE_TB, "Tuberculosis"),
        dest_dir=DEST_DIR,
        class_name="Tuberculosis",
        val_size=VAL_SIZE
    )

    # 4. Normal adicional (desde carpeta Normal del dataset TB, para tener más normales)
    # NOTA: Ya tenemos normales del dataset de neumonía, pero si quieres agregar más,
    # descomenta las siguientes líneas. Sino, omítelas para no duplicar.
    # copy_files(
    #     src_dir=os.path.join(SOURCE_TB, "Normal"),
    #     dest_dir=DEST_DIR,
    #     class_name="Normal",
    #     val_size=VAL_SIZE
    # )

    print(" .....¡Organización completada!")
    print(f" Los datos están en la carpeta '{DEST_DIR}'")