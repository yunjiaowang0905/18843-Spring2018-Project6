# Physics Model

Predicting air pollution concentrations with physics model.

## Dependencies

I used Python 2.7. All dependencies are listed in phy/requirements.txt

To install with pip: 

```
pip install -r requirements.txt
```

## Note

The current version only uses Gaussian Interpolation to fill in missing data in a 2D grid. More complex prediction methods will be added later. 

## Step 1: Convert Input CSV Data into .mat data

1. Download the CSV data file (distribution_Res.csv).

2. Create a directory where you want to save the matrix data (.mat file);

3. Change the file paths in csv2mat.py (marked with "TODO").

4. Run: 

```
python phy/tom/csv2mat.py
```

5. The matrix file will be saved as to the path you specified. 

6. You may want to inspect the .mat file to make sure data are properly filled in. Data is saved as a 3-d matrix named "data_all". The size of the matrix should be 5903 x 60 x 12 (T x n_lat x n_lon). Grids where data is missing has value "-1", same as the input CSV file.

## Step 2: Split Training and Testing Data

This step splits data into training set (90%) and test set (10%).

1. Change the file paths in separateDataset.py (marked with "TODO").

2. Run: 

```
python phy/tom/separateDataset.py
```

## Step 3: Fill in missing data in training set with 2D interpolation

1. Change the file paths in interpolate.py (marked with "TODO").

2. Run: 

```
python phy/tom/interpolate.py
```

3. Inspect the interpolated matrix data. The matrix should have the same shape as the original matrix. It should not contain any "-1", but may has some "0".  

## Step 4: Predict Concentration with Interpolated Training Data

This step takes about an hour to run all time slices. To run a subset of time slices, you can specify "time_range" parameter for function "predictSequence": 

```
data_pred = predictSequence(data_train, data_interp, time_range=(0, 100))
```

1. Change the file paths in predict.py (marked with "TODO").

2. Run: 

```
python phy/tom/predict.py
```

3. Prediction result will be saved to a .mat file (Line 186).