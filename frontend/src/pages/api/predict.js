# Backend API Implementation

import { fetchHistoricalData, predictFuture } from '../../utils/prediction';

export default async function handler(req, res) {
  const { date, location } = req.query;

  try {
    const historicalData = await fetchHistoricalData(date, location);
    const predictions = await predictFuture(historicalData);

    res.status(200).json({ predictions });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}
