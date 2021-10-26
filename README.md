# Person Re-identification

#### Requirements: Python=3.6 and Pytorch>=1.0.0

1. Install [Pytorch](http://pytorch.org/)

2. Download dataset (including Camstyle)

   - Market-1501 [[GoogleDriver]](https://drive.google.com/drive/folders/19aAOnGG8dZ3G5XwFq__YHlYBUQJFj4cO?usp=sharing)
   
   - DukeMTMC-reID [[GoogleDriver]](https://drive.google.com/drive/folders/1Bv25zdA3Hfzbx7WkzFxa7HaXDp9mA7Cx?usp=sharing)
   
   - MSMT17 [[GoogleDriver]](https://drive.google.com/drive/folders/11373fHJxxq8GESsxArZ8oJYXd-wLLWAK?usp=sharing)
   
   - Unzip each dataset and corresponding CamStyle under 'ECN/data/'
   
   Ensure the File structure is as follow:
   
   ```
   ECN/data    
   │
   └───market OR duke OR msmt17
      │   
      └───bounding_box_train
      │   
      └───bounding_box_test
      │   
      └───bounding_box_train_camstyle
      | 
      └───query
   ```

### Training and test domain adaptation model for person re-ID

  ```Shell
  # For Duke to Market-1501
  python main.py -s duke -t market --logs-dir logs/duke2market-ECN
  
  # For Market-1501 to Duke
  python main.py -s market -t duke --logs-dir logs/market2duke-ECN
  
  # For Market-1501 to MSMT17
  python main.py -s market -t msmt17 --logs-dir logs/market2msmt17-ECN --re 0
  
  # For Duke to MSMT17
  python main.py -s duke -t msmt17 --logs-dir logs/duke2msmt17-ECN --re 0
  ```
