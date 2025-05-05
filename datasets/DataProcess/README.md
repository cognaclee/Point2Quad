## Introduction

**Data Processing: Preprocessing and Postprocessing** (works on Windows)

## Usage

**PreProcess:**

1. Place the `.obj` files into the `data\0_obj\` directory, then run:
  
  ```bash
  bat\0_obj2m.bat
  ```
  
  The `.obj` files will be converted to `.m` files and saved in `data\1_m\`.
  
2. Run the preprocessing script:
  
  ```bash
  bat\1_pre.bat
  ```
  
  The preprocessed data will be stored in `data\2_pre\`.

**PostProcess:**
  
3. Run `Point2Quad` and save the generated `.m` files into the `data\3_pred\` directory.

4. Execute the following batch scripts in order as postprocess:
  
  ```bash
  bat\2_post1.bat
  bat\3_post2.bat
  bat\4_finaloutput_m.bat
  bat\5_finaloutput_obj.bat
  ```
  
  The final output `.m` and `.obj` files will be saved in:
  
  - `data\5_finaloutput_m\`
  - `data\6_finaloutput_obj\`
