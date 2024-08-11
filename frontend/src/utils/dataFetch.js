import { TimeSeriesDataSet } from 'pytorch_forecasting';

export function predictFuture(model, historicalData) {
    const validationDataset = TimeSeriesDataSet.fromDataset(
        train_dataset,  // Replace this with your train dataset
        historicalData,
        predict=true,
        stopRandomization=true
    );

    const valDataloader = validationDataset.toDataloader({ train: false, batchSize: 64, numWorkers: 4 });

    if (model.device === 'cuda') {
        model = model.to('cuda');
    }

    const predictions = model.predict(valDataloader);
    return predictions;
}
