
use linfa::prelude::*;
use linfa_svm::Svm;
use ndarray::{Array1, Array2};
use plotters::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

// --------------------------------------------------
// Configuration
// --------------------------------------------------

const SAMPLES_PER_CLASS: usize = 150;
const TRAIN_RATIO: f64 = 0.8;

// --------------------------------------------------
// Step 1: Generate a synthetic binary dataset
// --------------------------------------------------

fn generate_dataset() -> Vec<(f64, f64, bool)> {
    let mut rng = thread_rng();

    // Class 0: centred around (-1.5, -1.0)
    let normal_0_x = Normal::new(-1.5, 1.0).unwrap();
    let normal_0_y = Normal::new(-1.0, 1.0).unwrap();

    // Class 1: centred around (1.5, 1.0)
    let normal_1_x = Normal::new(1.5, 1.0).unwrap();
    let normal_1_y = Normal::new(1.0, 1.0).unwrap();

    let mut data = Vec::new();

    for _ in 0..SAMPLES_PER_CLASS {
        data.push((
            normal_0_x.sample(&mut rng),
            normal_0_y.sample(&mut rng),
            false,
        ));

        data.push((
            normal_1_x.sample(&mut rng),
            normal_1_y.sample(&mut rng),
            true,
        ));
    }

    data.shuffle(&mut rng);
    data
}

// --------------------------------------------------
// Step 2: Split into training and testing datasets
// Stratified split: preserve class proportions.
// --------------------------------------------------

fn train_test_split(
    data: &[(f64, f64, bool)],
) -> (
    Vec<(f64, f64, bool)>,
    Vec<(f64, f64, bool)>,
) {
    let mut class_0: Vec<_> = data
        .iter()
        .filter(|point| !point.2)
        .copied()
        .collect();

    let mut class_1: Vec<_> = data
        .iter()
        .filter(|point| point.2)
        .copied()
        .collect();

    let mut rng = thread_rng();

    class_0.shuffle(&mut rng);
    class_1.shuffle(&mut rng);

    let train_0 = (class_0.len() as f64 * TRAIN_RATIO) as usize;
    let train_1 = (class_1.len() as f64 * TRAIN_RATIO) as usize;

    let mut train = Vec::new();
    let mut test = Vec::new();

    train.extend_from_slice(&class_0[..train_0]);
    train.extend_from_slice(&class_1[..train_1]);

    test.extend_from_slice(&class_0[train_0..]);
    test.extend_from_slice(&class_1[train_1..]);

    train.shuffle(&mut rng);
    test.shuffle(&mut rng);

    (train, test)
}

// --------------------------------------------------
// Convert feature tuples into ndarray matrices
// --------------------------------------------------

fn extract_features(
    data: &[(f64, f64, bool)],
) -> (Array2<f64>, Array1<bool>) {
    let features: Vec<f64> = data
        .iter()
        .flat_map(|point| [point.0, point.1])
        .collect();

    let labels: Vec<bool> = data
        .iter()
        .map(|point| point.2)
        .collect();

    (
        Array2::from_shape_vec((data.len(), 2), features).unwrap(),
        Array1::from_vec(labels),
    )
}

// --------------------------------------------------
// Step 3: Standardise features
// Fit scaler using training data only.
// --------------------------------------------------

#[derive(Debug)]
struct StandardScaler {
    mean: [f64; 2],
    std: [f64; 2],
}

impl StandardScaler {
    fn fit(data: &Array2<f64>) -> Self {
        let n = data.nrows() as f64;

        let mean = [
            data.column(0).sum() / n,
            data.column(1).sum() / n,
        ];

        let std = [
            (data.column(0)
                .mapv(|x| (x - mean[0]).powi(2))
                .sum() / n)
                .sqrt(),

            (data.column(1)
                .mapv(|x| (x - mean[1]).powi(2))
                .sum() / n)
                .sqrt(),
        ];

        Self {
            mean,
            std: [
                if std[0] == 0.0 { 1.0 } else { std[0] },
                if std[1] == 0.0 { 1.0 } else { std[1] },
            ],
        }
    }

    fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut scaled = data.clone();

        for i in 0..data.nrows() {
            for j in 0..2 {
                scaled[[i, j]] =
                    (data[[i, j]] - self.mean[j]) / self.std[j];
            }
        }

        scaled
    }
}

// --------------------------------------------------
// Step 6: Evaluate classification performance
// --------------------------------------------------

fn evaluate(
    actual: &Array1<bool>,
    predicted: &Array1<bool>,
) {
    let mut tp = 0;
    let mut tn = 0;
    let mut fp = 0;
    let mut fn_count = 0;

    for (&actual, &predicted) in actual.iter().zip(predicted.iter()) {
        match (actual, predicted) {
            (true, true) => tp += 1,
            (false, false) => tn += 1,
            (false, true) => fp += 1,
            (true, false) => fn_count += 1,
        }
    }

    let total = actual.len() as f64;

    let accuracy = (tp + tn) as f64 / total;

    let precision = if tp + fp == 0 {
        0.0
    } else {
        tp as f64 / (tp + fp) as f64
    };

    let recall = if tp + fn_count == 0 {
        0.0
    } else {
        tp as f64 / (tp + fn_count) as f64
    };

    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    println!("\nClassification results");
    println!("----------------------");
    println!("Accuracy:  {:.4}", accuracy);
    println!("Precision: {:.4}", precision);
    println!("Recall:    {:.4}", recall);
    println!("F1-score:  {:.4}", f1);

    println!("\nConfusion matrix:");
    println!("                 Predicted");
    println!("                 False   True");
    println!("Actual False     {:5}   {:5}", tn, fp);
    println!("Actual True      {:5}   {:5}", fn_count, tp);
}

// --------------------------------------------------
// Step 7: Visualise decision regions
// --------------------------------------------------

fn visualize(
    model: &Svm<f64, bool>,
    scaler: &StandardScaler,
    train: &[(f64, f64, bool)],
    test: &[(f64, f64, bool)],
) -> Result<(), Box<dyn std::error::Error>> {
    // Transform observations into standardised coordinates.
    let (x_train, _) = extract_features(train);
    let (x_test, _) = extract_features(test);

    let x_train = scaler.transform(&x_train);
    let x_test = scaler.transform(&x_test);

    let x_min = x_train
        .column(0)
        .iter()
        .chain(x_test.column(0).iter())
        .fold(f64::INFINITY, |a, &b| a.min(b)) - 1.0;

    let x_max = x_train
        .column(0)
        .iter()
        .chain(x_test.column(0).iter())
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 1.0;

    let y_min = x_train
        .column(1)
        .iter()
        .chain(x_test.column(1).iter())
        .fold(f64::INFINITY, |a, &b| a.min(b)) - 1.0;

    let y_max = x_train
        .column(1)
        .iter()
        .chain(x_test.column(1).iter())
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 1.0;

    const GRID_SIZE: usize = 100;

    let mut grid_points = Vec::new();

    for row in 0..GRID_SIZE {
        for col in 0..GRID_SIZE {
            let x = x_min + (x_max - x_min) * col as f64
                / (GRID_SIZE - 1) as f64;

            let y = y_min + (y_max - y_min) * row as f64
                / (GRID_SIZE - 1) as f64;

            grid_points.push((x, y));
        }
    }

    let grid_values: Vec<f64> = grid_points
        .iter()
        .flat_map(|&(x, y)| [x, y])
        .collect();

    let grid = Array2::from_shape_vec(
        (grid_points.len(), 2),
        grid_values,
    )?;

    let predictions = model.predict(&grid);

    // Create the chart.
    let root = BitMapBackend::new("svm_decision_regions.png", (1000, 800))
        .into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Linear SVM Decision Regions", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(45)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Feature 1 (standardised)")
        .y_desc("Feature 2 (standardised)")
        .draw()?;

    // Draw decision regions using small rectangles.
    let dx = (x_max - x_min) / GRID_SIZE as f64;
    let dy = (y_max - y_min) / GRID_SIZE as f64;

    for (index, &(x, y)) in grid_points.iter().enumerate() {
        let colour = if predictions[index] {
            RGBColor(255, 150, 150).mix(0.35)
        } else {
            RGBColor(120, 170, 255).mix(0.35)
        };

        chart.draw_series(std::iter::once(Rectangle::new(
            [
                (x - dx / 2.0, y - dy / 2.0),
                (x + dx / 2.0, y + dy / 2.0),
            ],
            colour.filled(),
        )))?;
    }

    // Draw training observations.
    for (index, point) in train.iter().enumerate() {
        let colour = if point.2 { &RED } else { &BLUE };

        chart.draw_series(std::iter::once(Circle::new(
            (x_train[[index, 0]], x_train[[index, 1]]),
            4,
            colour.filled(),
        )))?;
    }

    // Draw testing observations.
    for (index, point) in test.iter().enumerate() {
        let colour = if point.2 { &RED } else { &BLUE };

        chart.draw_series(std::iter::once(Cross::new(
            (x_test[[index, 0]], x_test[[index, 1]]),
            7,
            colour.stroke_width(2),
        )))?;
    }

    root.present()?;

    println!("\nDecision-region graph saved as:");
    println!("svm_decision_regions.png");

    Ok(())
}

// --------------------------------------------------
// Main program
// --------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Generate dataset.
    let data = generate_dataset();

    // 2. Split into training and testing sets.
    let (train, test) = train_test_split(&data);

    let (x_train, y_train) = extract_features(&train);
    let (x_test, y_test) = extract_features(&test);

    println!("Training samples: {}", x_train.nrows());
    println!("Testing samples:  {}", x_test.nrows());

    // 3. Standardise features.
    let scaler = StandardScaler::fit(&x_train);

    let x_train = scaler.transform(&x_train);
    let x_test = scaler.transform(&x_test);

    // 4. Create and train linear SVM.
    let dataset = Dataset::new(x_train, y_train);

    let model = Svm::<f64, bool>::params()
        .linear_kernel()
        .pos_neg_weights(1.0, 1.0)
        .fit(&dataset)?;

    // 5. Make predictions.
    let predictions = model.predict(&x_test);

    // 6. Evaluate the model.
    evaluate(&y_test, &predictions);

    // 7. Visualise decision regions.
    visualize(&model, &scaler, &train, &test)?;

    Ok(())
}
