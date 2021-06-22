// // Load datasets API
// use smartcore::dataset::*;
// // DenseMatrix wrapper around Vec
// use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// // K-Means
// use smartcore::cluster::kmeans::{KMeans, KMeansParameters};
// // Performance metrics
// use smartcore::metrics::{homogeneity_score, completeness_score, v_measure_score};
//
// fn clustering() {
// // Load dataset
//     let digits_data = digits::load_dataset();
// // Transform dataset into a NxM matrix
//     let x = DenseMatrix::from_array(
//         digits_data.num_samples,
//         digits_data.num_features,
//         &digits_data.data,
//     );
// // These are our target class labels
//     let true_labels = digits_data.target;
// // Fit & predict
//     let labels = KMeans::fit(&x, KMeansParameters::default().with_k(10))
//         .and_then(|kmeans| kmeans.predict(&x))
//         .unwrap();
// // Measure performance
//     println!("Homogeneity: {}", homogeneity_score(&true_labels, &labels));
//     println!("Completeness: {}", completeness_score(&true_labels, &labels));
//     println!("V Measure: {}", v_measure_score(&true_labels, &labels));
// }