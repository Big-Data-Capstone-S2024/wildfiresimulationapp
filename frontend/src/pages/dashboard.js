import { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import Layout from '../components/Layout';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import { supabase } from '../utils/supabaseClient';

export default function Dashboard() {
  const [selectedDate, setSelectedDate] = useState(() => {
    return new Date(2023, 10, 1); // Set the initial date to January 1, 2024
  });
  const [location, setLocation] = useState('');
  const [email, setEmail] = useState('');
  const [mapSrc, setMapSrc] = useState('');
  const [historicalData, setHistoricalData] = useState([]); // State to store historical data
  const [predictions, setPredictions] = useState([]); // State to store predictions
  const router = useRouter();
  const googleMapsApiKey = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY;

  useEffect(() => {
    const checkUser = async () => {
      const { data: { session }, error } = await supabase.auth.getSession();

      if (error || !session) {
        router.push('/login');
      } else {
        setEmail(session.user.email);
      }
    };

    checkUser();

    const { lat, lng } = router.query;

    if (lat && lng) {
      setMapSrc(
        `https://www.google.com/maps/embed/v1/view?key=${googleMapsApiKey}&center=${lat},${lng}&zoom=12&markers=${lat},${lng}`
      );
    } else {
      setMapSrc(
        `https://www.google.com/maps/embed/v1/view?key=${googleMapsApiKey}&center=56.1304,-106.3468&zoom=4`
      );
    }
  }, [router, googleMapsApiKey, router.query]);

  useEffect(() => {
    const fetchHistoricalData = async () => {
      try {
        // Fetch historical data from the backend
        const response = await fetch(`/historical-data?date=${selectedDate.toISOString().split('T')[0]}`);
        
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        setHistoricalData(data.queried_data || []);
        console.log('Historical Data:', data);
      } catch (error) {
        console.error('Error fetching historical data:', error);
      }
    };

    fetchHistoricalData();
  }, [selectedDate]);

  useEffect(() => {
    const fetchPredictions = async () => {
      // Static data for the dashboard
      const staticPredictions = [
        { time_idx: 0, lon: -123.123, lat: 49.2827, cfb: 0.7, prediction: 1.23 },
        { time_idx: 1, lon: -123.124, lat: 49.2830, cfb: 0.6, prediction: 1.45 },
        { time_idx: 2, lon: -123.125, lat: 49.2833, cfb: 0.8, prediction: 1.67 },
        { time_idx: 3, lon: -123.126, lat: 49.2835, cfb: 0.9, prediction: 1.89 },
        // Add more data points as needed
      ];

      try {
        // Attempt to fetch predictions from the model_predict API
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ historical_data: historicalData }), // Send historical data to the API
        });

        if (!response.ok) {
          throw new Error('Network response was not ok');
        }

        const data = await response.json();

        if (data && data.predictions && data.predictions.length > 0) {
          setPredictions(data.predictions); // Use dynamic data if available
        } else {
          // If no predictions are returned, use static data
          setPredictions(staticPredictions);
        }
      } catch (error) {
        console.error('Error fetching predictions:', error);
        // If there's an error, use static data
        setPredictions(staticPredictions);
      }
    };

    fetchPredictions();
  }, [selectedDate, historicalData]);

  const handleDateChange = (date) => {
    setSelectedDate(date);
  };

  const handleLocationSearch = async (e) => {
    e.preventDefault();
    if (!location) {
      alert('Please enter a location.');
      return;
    }

    try {
      const response = await fetch(
        `https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(
          location
        )}&key=${googleMapsApiKey}`
      );
      const data = await response.json();

      if (data.status === 'OK') {
        const { lat, lng } = data.results[0].geometry.location;
        setMapSrc(
          `https://www.google.com/maps/embed/v1/view?key=${googleMapsApiKey}&center=${lat},${lng}&zoom=12&markers=${lat},${lng}`
        );
      } else {
        alert('Location not found. Please try again.');
      }
    } catch (error) {
      alert('An error occurred while fetching the location. Please try again.');
    }
  };

  return (
    <Layout>
      <div className="relative w-full h-full">
        <div className="date-picker-wrapper">
          <DatePicker
            selected={selectedDate}
            onChange={handleDateChange}
            className="px-3 py-2 border rounded"
            dateFormat="yyyy/MM/dd"
            maxDate={new Date()}
          />
        </div>
        <div className="map-container" style={{ position: 'relative', width: '100%', height: '500px' }}>
          <iframe
            src={mapSrc}
            width="100%"
            height="100%"
            frameBorder="0"
            style={{ border: 0 }}
            allowFullScreen=""
            aria-hidden="false"
            tabIndex="0"
          ></iframe>
          <div className="legend">
            <span className="low">Low</span>
            <span className="medium">Medium</span>
            <span className="high">High</span>
            <span className="very-high">Very High</span>
            <span className="extreme">Extreme</span>
          </div>
        </div>
        <div>
          <form onSubmit={handleLocationSearch} className="w-full flex items-center mt-4">
            <input
              type="text"
              className="flex-grow px-3 py-2 border rounded"
              placeholder="Search location"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
            />
            <button
              type="submit"
              className="ml-2 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
            >
              Search
            </button>
          </form>
        </div>
        <div>
          {/* Example of rendering markers or other data points on the map */}
          {predictions.map((prediction, index) => (
            <div key={index}>
              <p>Time: {prediction.time_idx}, Lat: {prediction.lat}, Lon: {prediction.lon}, CFB: {prediction.cfb}, Prediction: {prediction.prediction}</p>
              {/* You can use this data to add markers to Google Maps or other visualizations */}
            </div>
          ))}
        </div>
      </div>
    </Layout>
  );
}
