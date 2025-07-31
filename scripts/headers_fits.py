from astropy.io import fits
import argparse

def visualize_fits_header(fits_filepath):
    """
    Abre un archivo FITS y muestra las cabeceras de cada HDU.

    Args:
        fits_filepath (str): Ruta al archivo FITS.
    """
    try:
        with fits.open(fits_filepath) as hdul:
            print(f"Información del archivo FITS: {fits_filepath}\n")
            hdul.info()
            print("\n--- Cabeceras ---")
            for i, hdu in enumerate(hdul):
                print(f"\n--- HDU {i} ---")
                print(repr(hdu.header))
    except FileNotFoundError:
        print(f"Error: El archivo '{fits_filepath}' no fue encontrado.")
    except OSError as e:
        print(f"Error al abrir o leer el archivo FITS: {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualiza las cabeceras de un archivo FITS.")
    parser.add_argument("filepath", help="Ruta al archivo FITS.")
    
    args = parser.parse_args()
    
    visualize_fits_header(args.filepath)
