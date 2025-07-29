import tkinter as tk
from tkinter import filedialog, messagebox

def submit_form(root, variables, config, search_params):
    """Handles the submission of form data."""
    try:
        # Get values dynamically from the form
        search_params.clear()  # Clear previous values
        search_params.update({
            key: (var.get() if isinstance(var, tk.StringVar) else bool(var.get()))
            for key, var in variables.items()
        })
        print("Submitted values:", search_params)
        messagebox.showinfo("Submission", "Parameters submitted successfully!")
        root.quit()  # Exit the main loop
    except ValueError as e:
        messagebox.showerror("Invalid Input", f"Error: {e}. Please check your inputs.")

def browse_file(variables):
    """Opens a file dialog to select a file and updates the filenames variable."""
    filename = filedialog.askopenfilenames()
    variables["filenames"].set(filename)

def create_tooltip(widget, text, root):
    """Creates a tooltip for a widget."""
    tooltip = tk.Label(root, text=text, relief="solid", bg="yellow", font=("Arial", 8), padx=5, pady=5)
    tooltip.place_forget()

    def show_tooltip(event):
        widget_x = widget.winfo_rootx()
        widget_y = widget.winfo_rooty()
        tooltip.place(x=widget_x + 10, y=widget_y + widget.winfo_height() + 5)
        tooltip.lift()

    def hide_tooltip(event):
        tooltip.place_forget()

    widget.bind("<Enter>", show_tooltip)
    widget.bind("<Leave>", hide_tooltip)

def run_gui(search_params):
    """Runs the GUI interface to collect search parameters."""
    # Configuration for inputs and checkboxes
    config = [
        {"type": "input", "label": "Filenames", "key": "filenames", "default": "all", "help": "Path to the input file."},
        {"type": "input", "label": "Start DM", "key": "start_dm", "default": "0.0", "help": "Starting dispersion measure (DM)."},
        {"type": "input", "label": "Number of DMs", "key": "num_dms", "default": "1", "help": "Number of DMs to process."},
        {"type": "input", "label": "DM Step", "key": "dm_step", "default": "1.0", "help": "Step size for DM values."},
        {"type": "input", "label": "Downsample", "key": "downsample", "default": "1", "help": "Downsampling factor."},
        {"type": "input", "label": "Number of Subbands", "key": "nsub", "default": "1", "help": "Number of subbands to use."},
        {"type": "input", "label": "Max width (m)", "key": "m", "default": "0.05", "help": "Maximum width of boxcar to search."},
        {"type": "input", "label": "Sigma Threshold", "key": "sigma", "default": "7.0", "help": "Detection threshold in sigma."},
        {"type": "input", "label": "dt", "key": "dt", "default": "0.5", "help": "Time difference to discriminate duplicates."},
        {"type": "input", "label": "Plot Window (s)", "key": "plot_window", "default": "0.5", "help": "Plotting window size in seconds."},
        {"type": "checkbox", "label": "Timeseries", "key": "timeseries", "default": False, "help": "Plot timeseries."},
        {"type": "checkbox", "label": "Spectral Average", "key": "spectral_average", "default": False, "help": "Plot time average frequency spectra."},
        {"type": "input", "label": "Number to Plot", "key": "num_to_plot", "default": "150", "help": "Number of candidates to plot."},
        {"type": "input", "label": "Stokes Parameter", "key": "stokes", "default": "0", "help": "Stokes parameter for analysis."},
        {"type": "checkbox", "label": "Save", "key": "save", "default": False, "help": "Save candidate plots in pdf file."},
        {"type": "checkbox", "label": "Convert to FIL", "key": "convert_to_fil", "default": False, "help": "Convert to filterbank format (for ARTE logfiles)."},
        {"type": "input", "label": "Bandpass (MHz)", "key": "bandpass", "default": "1200,1800", "help": "Frequency range of the bandpass."},
        {"type": "input", "label": "Time Resolution (s)", "key": "time_resolution", "default": "0.01", "help": "Time resolution in seconds."},
        {"type": "input", "label": "Number of Channels", "key": "nchannels", "default": "2048", "help": "Number of frequency channels."},
        {"type": "checkbox", "label": "Flip Data", "key": "flip_data", "default": False, "help": "Flip frequency channels."},
        {"type": "input", "label": "Padding", "key": "padding", "default": "0", "help": "Add data to increase time of file."},
        {"type": "input", "label": "Remove Samples", "key": "remove_samples", "default": "0", "help": "Number of samples to remove at the beginning of the file."},
        {"type": "checkbox", "label": "Remove Filterbank Files", "key": "remove_fil", "default": False, "help": "Remove filterbank files after processing."},
        {"type": "checkbox", "label": "Run Prepdata", "key": "prepdata", "default": False, "help": "Run prepdata command."},
        {"type": "checkbox", "label": "Wrap PDFs", "key": "wrap_pdfs", "default": False, "help": "Wrap PDFs into a single file."},
        {"type": "input", "label": "Output Directory", "key": "output_dir", "default": "./", "help": "Directory to save output files."},
        {"type": "input", "label": "Logfiles Directory", "key": "logfiles_dir", "default": "./", "help": "Directory where logfiles are stored."},
        {"type": "checkbox", "label": "Use RFI Mask", "key": "mask", "default": False, "help": "Use previously generated rfifind RFI mask."},
        {"type": "checkbox", "label": "Remove PRESTO Files", "key": "remove_presto", "default": False, "help": "Remove PRESTO files after processing."},
        {"type": "input", "label": "Ncpus", "key": "ncpus", "default": "1", "help": "Number of CPUs to use for processing."},
        {"type": "input", "label": "Rfifind time (s)", "key": "rfifind_time", "default": "2", "help": "Time for rfifind processing."},
        {"type": "input", "label": "Clip Presto", "key": "clip_presto", "default": "0.0", "help": "Clip value for PRESTO."},
        {"type": "input", "label": "Container type", "key": "container", "default": "singularity", "help": "Container system to use: Singularity, Docker, or internal execution."}]

    # Set up the main window
    root = tk.Tk()
    root.title("FRB Search Parameters")

    # Variable storage
    variables = {}

    # Dynamically create UI elements
    for row, item in enumerate(config):
        if item["type"] == "input":
            variables[item["key"]] = tk.StringVar()
        elif item["type"] == "checkbox":
            variables[item["key"]] = tk.BooleanVar()
        elif item["type"] == "dropdown":
            variables[item["key"]] = tk.StringVar()
        else:
            raise ValueError(f"Unknown input type: {item['type']}")
        variables[item["key"]].set(item["default"])

        # Create label and tooltip
        label = tk.Label(root, text=item["label"], fg="black", cursor="hand2")
        label.grid(row=row, column=0)
        create_tooltip(label, item["help"], root)

        # Create input or checkbox
        if item["type"] == "input":
            entry = tk.Entry(root, textvariable=variables[item["key"]])
            entry.grid(row=row, column=1)
            if item["key"] == "filenames":
                browse_button = tk.Button(root, text="Browse", command=lambda: browse_file(variables))
                browse_button.grid(row=row, column=2)
        elif item["type"] == "checkbox":
            checkbox = tk.Checkbutton(root, variable=variables[item["key"]])
            checkbox.grid(row=row, column=1)

    # Add Submit and Clear buttons
    submit_button = tk.Button(root, text="Submit", command=lambda: submit_form(root, variables, config, search_params))
    submit_button.grid(row=len(config), column=1)

    clear_button = tk.Button(
        root, text="Clear", command=lambda: [var.set(item["default"]) for item, var in zip(config, variables.values())]
    )
    clear_button.grid(row=len(config), column=2)

    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit without submitting?"):
            search_params.clear()  # Clear search_params if the window is closed
            root.quit()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.mainloop()

def main():
    """Main function to execute the GUI."""
    search_params = {}  # This will be updated by run_gui
    run_gui(search_params)
    print("Search Parameters:", search_params)

if __name__ == "__main__":
    main()