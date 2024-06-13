import os
import xml.etree.ElementTree as ET

def load_xml_files(file, loggs_enabled=False):

    path = f'/Users/luis.caporal/Documents/Notebooks/DS_TEST/DS_Project/data/{file}'
    data = []
    files = []
    not_XML_valid = []
    without_summary = []

    for index, file in enumerate(os.listdir(path)):
        if file.endswith('.xml'):
            file_path = os.path.join(path, file)
            try:
                if os.path.getsize(file_path) == 0:
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as archivo:
                    contenido = archivo.read()
                    if not contenido.strip().startswith('<?xml'):
                        not_XML_valid.append(file)
                        continue

                    tree = ET.ElementTree(ET.fromstring(contenido))
                    root = tree.getroot()
                    for award in root.findall('.//Award'):
                        for element in award:
                            if element.tag == 'AbstractNarration':
                                summary = element.text
                                
                if summary is not None:
                    data.append(summary)
                    files.append(file)
                else:
                    without_summary.append(file)
            except ET.ParseError as e:
                print(f"Error de parseo en {file_path}: {e}")
            except Exception as e:
                print(f"Error inesperado con {file_path}: {e}")

    if loggs_enabled:
        if len(data) == index + 1 and loggs_enabled:
            print('All files were imported successfully.')
        else:
            print(f'{len(data)} files imported successfully.')
            print(f'There was an error importing {(index + 1) - len(data)} files:')
            
            print('')
            print(f'{len(not_XML_valid)} damaged files: {not_XML_valid[0]}, {not_XML_valid[1]}, {not_XML_valid[2]}')
            print(f'{len(without_summary)} files without summary: {without_summary[0]}, {without_summary[1]}, {without_summary[2]}, ...')
    return data, files, not_XML_valid, without_summary

if __name__ == '__main__':
    data, _, _, _ = load_xml_files('2020', loggs_enabled=True)
