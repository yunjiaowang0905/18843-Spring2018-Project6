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

3. You most likely want to change the file paths in csv2mat.py (marked with "TODO").

4. Run: 

```
python phy/tom/csv2mat.py
```

5. The matrix file will be saved as to the path you specified. 

6. You may want to inspect the .mat file to make sure data are properly filled in. Data is saved as a 3-d matrix named "data_all". The size of the matrix should be 5942 x 73 x 18 (T x n_lat x n_lon). Grids where data is missing has value "-1", same as the input CSV file.

## Step 2: Fill in missing data with 2D interpolation

1. Same as previous step, you probably want to change the file paths in interpolate.py (marked with "TODO").

2. Run: 

```
python phy/tom/interpolate.py
```

3. Inspect the interpolated matrix data. The matrix should have the same shape as the original matrix. It should not contain any "-1", but may has some "0".  