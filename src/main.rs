use nalgebra::{DMatrix, Scalar};
use std::time::Instant;
use std::io::prelude::*;
use std::io::BufReader;
use std::fs::File;
use std::str::FromStr;
use smartcore::metrics::mean_squared_error;
use smartcore::metrics::r2;
use smartcore::linear::linear_regression::{LinearRegression, LinearRegressionParameters, LinearRegressionSolverName};

use smartcore::neighbors::knn_classifier::KNNClassifier;
use smartcore::model_selection::train_test_split;
use smartcore::metrics::mean_absolute_error;
use csv::StringRecord;
// fn read_csv(path: &str)  -> (usize,usize, Vec<f64>, Vec<String>) {
// fn read_csv(path: &str)  {
// fn read_csv(path: &str)  -> (Result<DenseMatrix<f32>, Box<dyn std::error::Error>>, Vec<String>, Vec<f32>) {
// fn read_csv(path: &str)  -> (Result<DMatrix<f32>, Box<dyn std::error::Error>>, Vec<String>, Vec<f32>) {
fn read_csv(path: &str)  -> (Result<Vec<f32>, Box<dyn std::error::Error>>, Vec<String>, Vec<f32>, usize, usize) {
// fn read_csv(path: &str)  -> (Result<(), Box<dyn std::error::Error>>, &StringRecord) {
// fn read_csv(path: &str)  -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = csv::Reader::from_path(path).unwrap();
    let headers2 = &reader.headers().unwrap();
    // println!("{:?}", headers2);
    let mut headers:Vec<String> = vec![];
    for i in headers2.into_iter() {
        headers.push(i.to_string());
    }
    println!("{:?}", headers);

    let mut rows = 0;
    let mut data = vec![];
    let mut target = vec![];
    // for result in reader.records() {
    for result in reader.deserialize() {
        rows += 1;
        let record:Vec<f32> = result.unwrap();
        // println!("{:?}", record);
        let len = record.len();
        let mut count = 0;
        for i in record {

            if count < len - 1 {
                data.push(i);
            }
            else {
                target.push(i);
            }
            count += 1;
        }
        // data.push(record);
    }
    // println!("{:?}", data);
    let cols = data.len()/ rows;
    // println!("{:?} target" , target);
    // (Ok(()), headers)
    // println!("{:#?} matrix", DMatrix::from_row_slice(rows, cols, &data[..]));
    // println!("{:#?} matrix array", DenseMatrix::from_array(rows, cols, &data[..]));

    // (Ok(DenseMatrix::from_array(rows, cols, &data[..])), headers, target)
    // (Ok(DMatrix::from_row_slice(rows, cols, &data[..])), headers, target)
    (Ok(data), headers, target, rows, cols)

    // (Ok((rows, cols, &data[..])), headers)
    // Ok(())
    //     (rows, cols, data, headers)
}

fn parse_csv<N, R>(input: R) -> Result<DMatrix<N>, Box<dyn std::error::Error>>
    where N: FromStr + Scalar,
          N::Err: std::error::Error,
          R: BufRead
{
    // initialize an empty vector to fill with numbers
    let mut data = Vec::new();

    // initialize the number of rows to zero; we'll increment this
    // every time we encounter a newline in the input
    let mut rows = 0;

    // for each line in the input,
    for line in input.lines() {
        // increment the number of rows
        rows += 1;
        // iterate over the items in the row, separated by commas
        for datum in line?.split_terminator(",") {
            // trim the whitespace from the item, parse it, and push it to
            // the data array
            data.push(N::from_str(datum.trim())?);
        }
    }

    // The number of items divided by the number of rows equals the
    // number of columns.
    let cols = data.len() / rows;

    // Construct a `DMatrix` from the data in the vector.
    Ok(DMatrix::from_row_slice(rows, cols, &data[..]))
}

// ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
// ["price"]
//["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat", "medv"]
use smartcore::dataset::*;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
fn linear_regr(path: &str){
// fn linear_regr(){
//     let (file, headers, target)  = read_csv(path);
    let (file, headers, target, rows, cols)  = read_csv(path);

    // let boston = DMatrix::from_row_slice(rows, cols, &data[..]);
    let file_data = file.unwrap();

    let dataset_file = Dataset {
        // data: file_data.data().as_vec().to_vec(), // for dmatrix
        data: file_data,
        target,
        // num_samples: file_data.nrows(),
        num_samples: rows,
        // num_features: file_data.ncols(),
        num_features: cols,
        feature_names: headers[0..headers.len() - 1].to_owned(),
        target_names: vec![headers.last().unwrap().clone()],
        description: "trying some stuff".to_string(),
    };
    // println!("{:?} feature_names", dataset_file.feature_names);
    // println!("{:?} file-data", dataset_file.data);
    println!("{:?} file-data", dataset_file);
    println!("dataset_ready");
    let x = DenseMatrix::from_array(
        dataset_file.num_samples, dataset_file.num_features, &dataset_file.data);
    println!("after x");
    let y = dataset_file.target;
    println!("after y");


//     let boston = boston::load_dataset();
//         let x = DenseMatrix::from_array(
//         boston.num_samples,
//         boston.num_features,
//         &boston.data,
//     );
// // These are our target class labels
//     let y = boston.target;
    println!("{:?} target len", y.len());
    // // Split dataset into training/test (80%/20%)
    let (x_train, x_test, y_train, y_test)
        = train_test_split(&x, &y, 0.2, true);
    println!("after train");
    println!("{:?} xtrain", x_train.shape()); // 8,13
    println!("{:?} xtest", x_test.shape());  // 2,13
    println!("{:?} ytrain", y_train.len()); //8
    println!("{:?} ytest", y_test.len());  //2

    let y_hat_lr = LinearRegression::fit(&x_train, &y_train, Default::default())
        .and_then(|lr| lr.predict(&x_test)).unwrap();

// Calculate test error
        println!("MSE: {}", mean_squared_error(&y_test, &y_hat_lr));
// //
    use smartcore::linalg::naive::dense_matrix::*;
    use smartcore::linear::linear_regression::*;
    let lin_reg = LinearRegression::fit(&x, &y,
                                   LinearRegressionParameters::default().
                                       with_solver(LinearRegressionSolverName::QR)).unwrap();
//
    let y_hat_lin = lin_reg.predict(&x).unwrap();
    println!("y_hat: {:?}", y_hat_lin);
    println!("intercept: {:?}", lin_reg.intercept());
    println!("coeff: {:?}", lin_reg.coefficients());
    // println!("coeff: {:?}", lin_reg.coefficients().values());
    let coeff =  lin_reg.coefficients().clone().to_row_vector();
    println!("{:?} cf", coeff);
    println!("MSE: {}", mean_squared_error(&y_test, &y_hat_lr));
    println!("mae: {}", mean_absolute_error(&y_test, &y_hat_lr));
    println!("r2: {}", r2(&y_test, &y_hat_lr));

    // let boston = boston::load_dataset();
    // println!("{:?}", boston.num_features);
    // println!("{:?}", boston.num_samples);
    // println!("{:?}", boston.target); // last column of values
    // println!("{:?}", boston.num_features);
    // println!("{:?}", boston.feature_names);
    // println!("{:?}", boston.target_names); // headers
    // println!("{:?}", boston.description);
//     println!("{:?} boston", boston);
// // Transform dataset into a NxM matrix
//     let x = DenseMatrix::from_array(
//         boston.num_samples,
//         boston.num_features,
//         &boston.data,
//     );
// // These are our target class labels
//     let y = boston.target;
// // Split dataset into training/test (80%/20%)
//     let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
// // Linear Regression
//     let y_hat_lr = LinearRegression::fit(&x_train, &y_train, Default::default())
//         .and_then(|lr| lr.predict(&x_test)).unwrap();
// // Calculate test error
//
//     println!("MSE: {}", mean_squared_error(&y_test, &y_hat_lr));
// //
//     use smartcore::linalg::naive::dense_matrix::*;
//     use smartcore::linear::linear_regression::*;
//     let lin_reg = LinearRegression::fit(&x, &y,
//                                    LinearRegressionParameters::default().
//                                        with_solver(LinearRegressionSolverName::QR)).unwrap();
//
//     let y_hat_lin = lin_reg.predict(&x).unwrap();
//     println!("y_hat: {:?}", y_hat_lin);
//     println!("intercept: {:?}", lin_reg.intercept());
//     println!("coeff: {:?}", lin_reg.coefficients());


    // let log_reg = LogisticRegression::fit(&x, &y, Default::default()).unwrap();
    // println!("log");
    // let y_hat_log = log_reg.predict(&x).unwrap();
    // println!("y_hat: {:?}", y_hat_log);

}

// fn linear_regr() {
//     let time = Instant::now();
//     let file = File::open("dataset.csv").unwrap();
//     // let file = File::open("boston_housing.csv").unwrap();
//     let bos: DMatrix<f64> = parse_csv(BufReader::new(file)).unwrap();
//     println!("{}", bos.rows(0, 5));
//
//     let x = bos.columns(0, 13).into_owned();
//     let y = bos.column(13).into_owned();
//
//     // let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y.transpose(), 0.2, false);
//     // 4 param 0.2.0 (4 -shuffle)
//     // let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y.transpose(), 0.2 );  // 3 param for 0.1.0
//     let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, false );  // 3 param for 0.1.0
//     let a = x_train.clone().insert_column(13, 1.0).into_owned();
//     let b = y_train.clone().transpose();
//
//     // np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
//     let x = (a.transpose() * &a).try_inverse().unwrap() * &a.transpose() * &b;
//     let coeff = x.rows(0, 13);
//     let intercept = x[(13, 0)];
//
//     println!("coeff: {}, intercept: {}", coeff, intercept);
//
//     // Q, R = np.linalg.qr(A)
//     let qr = a.qr();
//     let (q, r) = (qr.q().transpose(), qr.r());
//     let x = r.try_inverse().unwrap() * &q * &b;
//     let coeff = x.rows(0, 13);
//     let intercept = x[(13, 0)];
//
//     println!("coeff: {}, intercept: {}", coeff, intercept);
//
//     let y_hat = (x_test * &coeff).add_scalar(intercept);
//
//     println!("mae: {}", mean_absolute_error(&y_test, &y_hat.transpose()));
//     println!("mse: {}", mean_absolute_error(&y_test, &y_hat.transpose()));
//     println!("r2: {}", r2(&y_test, &y_hat.transpose()));
//     let time = time.elapsed().as_millis();
//     println!("time {}", time)
// }

// Load datasets API
// use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
// K-Means
use smartcore::cluster::kmeans::{KMeans, KMeansParameters};
// Performance metrics
use smartcore::metrics::{homogeneity_score, completeness_score, v_measure_score};
use smartcore::dataset::Dataset;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::linalg::BaseMatrix;
use std::io;

//
// https://docs.rs/smartcore/0.1.0/smartcore/dataset/diabetes/index.html   datasets
fn clustering() {
// Load dataset
//     let digits_data = digits::load_dataset();
    // let digits_data = diabetes::load_dataset();
    let digits_data = Dataset {
        data: [ 0.0, 2.0, 12.0, 10.0, 16.0, 5.0, 0.0, 0.0, 0.0, 11.0, 21.0, 3.0, 2.0, 0.0].to_vec(),
        target: [0.0, 9.0, 15.0, 2.0].to_vec(),
    //     // num_samplestarget: vec![],
        num_samples: 4,
        num_features: 3,
        feature_names: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        target_names: vec!["one".to_string(), "two".to_string(), "three".to_string(), "four".to_string()],
    // , "five".to_string(),
    //                        "six".to_string(), "seven".to_string(), "eight".to_string(), "nine".to_string(), "ten".to_string()],
        description: "trying some stuff".to_string(),
    };
    println!("{:?} dataset", digits_data);
    println!("{:?} dataset", digits_data.data.len());
    println!("{:?} dataset.target", digits_data.target.len());
    println!("{:?} dataset.num_samples", digits_data.num_samples);
    println!("{:?} dataset.num_features", digits_data.num_features);
    println!("{:?} dataset.feature_names", digits_data.feature_names);
    println!("{:?} dataset.target_names", digits_data.target_names);
    println!("{:?} dataset.description", digits_data.description);
// Transform dataset into a NxM matrix
    println!("0");
    let x = DenseMatrix::from_array(
        digits_data.num_samples,
        digits_data.num_features,
        &digits_data.data,
    );
    println!("1");
// These are our target class labels
    let true_labels = digits_data.target;
    println!("2");
// Fit & predict
    let labels = KMeans::fit(&x, KMeansParameters::default().with_k(10)) // 2 param for 0.2.0 or 3 for 0.1.0
    // let labels = KMeans::fit(&x,10, KMeansParameters::default()) // 2 param for 0.2.0 or 3 for 0.1.0 and no `with_k(10) -instead k -clusters arg
        .and_then(|kmeans| kmeans.predict(&x))
        .unwrap();
// Measure performance
    println!("Homogeneity: {}", homogeneity_score(&true_labels, &labels));
    println!("Completeness: {}", completeness_score(&true_labels, &labels));
    println!("V Measure: {}", v_measure_score(&true_labels, &labels));
}
pub fn test_boston() {
// Load dataset
    let boston = boston::load_dataset();
    println!("{:?}", &boston);


// Transform dataset into a NxM matrix
    let x = DenseMatrix::from_array(
        boston.num_samples,
        boston.num_features,
        &boston.data,
    );
// These are our target class labels
    let y = boston.target;
// Split dataset into training/test (80%/20%)
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    // let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2);
// Linear Regression
    let y_hat_lr = LinearRegression::fit(&x_train, &y_train, Default::default())
        .and_then(|lr| lr.predict(&x_test)).unwrap();
    let lin_reg = LinearRegression::fit(&x, &y,
                                        LinearRegressionParameters::default().
                                            with_solver(LinearRegressionSolverName::QR)).unwrap();
// Calculate test error
    println!("MSE: {}", mean_squared_error(&y_test, &y_hat_lr));
    println!("mae: {}", mean_absolute_error(&y_test, &y_hat_lr));
    println!("r2: {}", r2(&y_test, &y_hat_lr));
    println!("coef: {}", &lin_reg.coefficients());
    println!("intercept: {}", &lin_reg.intercept());
}
// https://cheesyprogrammer.com/2018/12/13/simple-linear-regression-from-scratch-in-rust/
fn main() {
    // linear_regr();
    // clustering();
    // read_csv("boston_housing2.csv");
    linear_regr("boston_housing2.csv");
    // linear_regr();
    // test_boston();
    // linear_regr("boston_housing3.csv");
}