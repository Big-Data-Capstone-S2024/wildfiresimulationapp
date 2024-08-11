import { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import Layout from '../components/Layout';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import Papa from 'papaparse'; // Import PapaParse to read CSV file

export default function Dashboard() {
  const [selectedDate, setSelectedDate] = useState(new Date('2024-01-30'));
  const [location, setLocation] = useState('');
  const [mapSrc, setMapSrc] = useState('');
  const [predictions, setPredictions] = useState([]); // State for predictions
  const router = useRouter();
  const googleMapsApiKey = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY;

  useEffect(() => {
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
  }, [googleMapsApiKey, router.query]);

  const handleDateChange = (date) => {
    setSelectedDate(date);
    fetchPredictions(date);
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

  const fetchPredictions = (date) => {
    Papa.parse('/historical_07_12.csv', {
        download: true,
        header: true,
        complete: (result) => {
            console.log(result.data); // Debugging line
            const filteredData = result.data.filter((row) => {
                const rowDate = new Date(`${row.year}-${row.month}-${row.day}`);
                return rowDate >= date && rowDate <= new Date(date.getTime() + 6 * 24 * 60 * 60 * 1000);
            });
            setPredictions(filteredData);
        },
    });
};


  return (
    <Layout>
      <div className="relative w-full h-full max-w-5xl max-h-96 mb-4">
        <div className="date-picker-wrapper">
          <DatePicker
            selected={selectedDate}
            onChange={handleDateChange}
            className="px-3 py-2 border rounded"
            dateFormat="yyyy/MM/dd"
            maxDate={new Date('2024-01-30')}
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
      </div>
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
      
      {/* Prediction Dashboard */}
      <div className="prediction-dashboard mt-8">
        <h2>7-Day Wildfire Forecast</h2>
        <ul>
          {predictions.map((prediction, index) => (
            <li key={index}>
              {prediction.date}: {prediction.location} - CFB: {prediction.cfb}
            </li>
          ))}
        </ul>
      </div>
    </Layout>
  );
}
