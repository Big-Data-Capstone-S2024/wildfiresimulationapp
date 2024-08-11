import { useEffect, useState } from 'react';
import { supabase } from '../utils/supabaseClient';
import { useRouter } from 'next/router';
import React from 'react';
import './YourStylesheet.css'; // Adjust this path based on where your CSS file is located

// Signup Component
export default function Signup() {
  return (
    <div className="form-container">
      <input type="email" placeholder="Email Address" />
      <input type="text" placeholder="Location" />
      <button>Sign Up</button>
      <div className="additional-text">
        if you have an existing account, please <a href="/login">login</a> here.
      </div>
    </div>
  );
}

// Home Component (assuming this should be in another file)
export function Home() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkSession = async () => {
      try {
        const { data: { session }, error } = await supabase.auth.getSession();
        if (error) throw error;

        if (session) {
          router.push('/NotificationPrompt');
        } else {
          router.push('/login');
        }
      } catch (error) {
        console.error('Error checking session:', error.message);
        router.push('/login');
      } finally {
        setLoading(false);
      }
    };

    checkSession();
  }, [router]);

  if (loading) return <div>Loading...</div>;

  return null; // 
}
