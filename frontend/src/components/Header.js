import { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import { supabase } from '../utils/supabaseClient';

export default function Header() {
  const [Email, setEmail] = useState('');
  const router = useRouter();

  useEffect(() => {
    const checkUser = async () => {
      const { data: { session } } = await supabase.auth.getSession();

      if (!session) {
        router.push('/login');
      } else {
        const user = session.user;
        setEmail(user.email);
      }
    };

    checkUser();
  }, [router]);

  const handleLogout = async () => {
    await supabase.auth.signOut();

    // Call API to clear session from cookies
    await fetch('/api/clearSession', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    router.push('/login'); // Redirect to login page
  };

  return (
    <header className="flex items-center justify-between p-4 bg-white shadow-md">
      <div className="flex items-center">
        <img
          src="/assets/logo.png"
          alt="Wildfire Simulation Logo"
          className="w-16 h-auto mr-4"
        />
        <h1 className="text-xl font-semibold">Wildfire Simulation</h1>
      </div>
      {Email && (
        <div className="flex items-center ml-auto text-gray-600 text-right">
          <span className="mr-4">{Email}</span>
          <button
            onClick={handleLogout}
            className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
          >
            Logout
          </button>
        </div>
      )}
    </header>
  );
}
