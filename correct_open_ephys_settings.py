from pathlib import Path
from packaging.version import parse


def correct_open_ephys_settings(open_ephys_folders, missing_channel="CH191",
                                missing_xpos=43, missing_ypos=1900):
    """
    Corrects settings for Open Ephys Neuropix

    Parameters
    ----------
    open_ephys_folders : str, Path, or list
        The open ephys folder(s) to correct
    """
    import xml.etree.ElementTree as ET

    
    if isinstance(open_ephys_folders, (str, Path)):
        open_ephys_folders = [open_ephys_folders]
        
    for folder in open_ephys_folders:
        print(f"Correcting folder: {folder}")
        folder = Path(folder).absolute()
        assert folder.is_dir()

        # find settings
        settings_files = [p for p in folder.iterdir() if p.suffix == ".xml" and "setting" in p.name]
        if len(settings_files) > 1:
            assert settings_file is not None, ("More than one settings file found. Specify a settings "
                                            "file with the 'settings_file' argument")
        elif len(settings_files) == 0:
            raise FileNotFoundError("Cannot find settings.xml file!")
        else:
            settings_file = settings_files[0]

        # parse xml
        tree = ET.parse(str(settings_file))
        root = tree.getroot()

        npix = None
        for child in root:
            if "SIGNALCHAIN" in child.tag:
                for proc in child:
                    if "PROCESSOR" == proc.tag:
                        name = proc.attrib["name"]
                        if name == "Neuropix-PXI":
                            npix = proc
                            break
        if npix is None:
            raise Exception("Can only correct settings when the Neuropix-PXI plugin is used")
        
        if parse(npix.attrib["libraryVersion"]) < parse("0.3.3"):
            raise Exception("Electrode locations are available from Neuropix-PXI version 0.3.3")


        for child in npix:
            if child.tag == "EDITOR":
                for child2 in child:
                    if child2.tag == "NP_PROBE":
                        np_probe = child2
                        break
        if np_probe is None: 
            raise Exception("Can't find 'NP_PROBE' information. Positions not available")

        # read channels
        needs_correction = False
        for child in np_probe:
            if child.tag == "CHANNELS":
                # add missing channel
                if missing_channel not in child.attrib:
                    child.attrib[missing_channel] = "0"
                    channels = child.attrib
                    needs_correction = True

        # read positions
        for child in np_probe:
            if child.tag == "ELECTRODE_XPOS":
                if missing_channel not in child.attrib:
                    child.attrib[missing_channel] = str(missing_xpos)
                    needs_correction = True
                    
                xpos = child.attrib
            if child.tag == "ELECTRODE_YPOS":
                if missing_channel not in child.attrib:
                    child.attrib[missing_channel] = str(missing_ypos)
                    needs_correction = True
                ypos = child.attrib
        
        if needs_correction:
            settings_file_path = str(settings_file)
            wrong_settings_file = settings_file.parent / "settings.xml.wrong"
            print(f"Renaming wrong settings file as {wrong_settings_file}")
            settings_file.rename(wrong_settings_file)
            print(f"Saving correct settings file as {settings_file_path}")
            tree.write(settings_file_path)
        else:
            print("No correction needed")
    
    