# Warning
I still have to better organize my code and better explain what and why I did certain things. I will publish soon a detailed explaination. However, most of the project is already explained in the project I submitted for one of my MSc exams you can find in the bottom of this README file

# Code Structure

The main entry point of the project is _main.ipynb_

This script orchestrates the execution of the pipeline and connects all core components of the project.

---

# Project Modules

The project is organized into modular components. Some internal utility modules are included in `./utils/`

## Utils
     
These modules contain supporting functions and internal logic required for the pipeline to run correctly. While they are not the main entry point, they are essential dependencies of the project.


## Data Folder

All datasets must be placed inside the following directory: `./data/`


Make sure the expected files and subfolders are available before running the project. Missing data may cause the pipeline to fail.

---

# Repository

GitHub repository from which I have taken inspiration about **ImpCHI**:  
👉 https://github.com/Gianbattistabsn/Winter-project-DSMLL-2025-2026-ImpChi-and-W2V