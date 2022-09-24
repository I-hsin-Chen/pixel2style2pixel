* * # Organ switch : stylegan 五官更換器
    
    * Our work is mainly in directory "organ_switch" and "reconstruction.py"

    ## Example

    #### Sources
    | Target face                          | Target eyes                          |
    | ------------------------------------ | ------------------------------------ |
    | ![](https://i.imgur.com/DDpA1WE.jpg) | ![](https://i.imgur.com/nXx1Klt.jpg) |
    | **Target mouth**                     | **Target nose**                      |
    | ![](https://i.imgur.com/EuLFjfk.jpg) | ![](https://i.imgur.com/HY91e6x.jpg) |
  
  
    #### Result
  
    | 2D reconstruction                    | 3D reconstruction                    |
    | ------------------------------------ | ------------------------------------ |
    | ![](https://i.imgur.com/XIBY50t.jpg) | ![](https://i.imgur.com/TbB2L0P.png) |
  
    
  
    ---
    ## Environment settings
  
    #### Install conda environment : 
    ```
    conda env create -f environment_organ_switch.yml
    ```
  
    #### Download pretrained model from pixel2style2pixel : 
  
    ```
    cd pixel2style2pixel
    wget --load-cookies /tmp/cookies.txt \
    "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc? \
    export=download&id=1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0" -O \
    /path/to/directory/pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt && rm -rf /tmp/cookies.txt
    ```
    * Remember to change the last line : `/path/to/directory/pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt` 
    to the path of your psp directory.
  
    #### Settings about gpu and cuda please refer to : [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)
  
    ---
  
    ## Run organ switch
    * change the string `path` in `reconstruction.py` to your own target image directory.
  
  
    ```
    python reconstruction.py
    ```
  
    * The images in your directory should be strictly named as `face`、`nose`、`mouth`、`nose`. 
    * Different file extensions are allowed.
    * Result will be generated at : `path/result` .
