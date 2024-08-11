import { useEffect, useState } from 'react';
import { supabase } from '../utils/supabaseClient';
import { useRouter } from 'next/router';

export default function Home() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkSession = async () => {
      try {
        const { data: { session }, error } = await supabase.auth.getSession();
        if (error) throw error;
        
        if (session) {
          router.push('/dashboard');
        } else {
          router.push('/signup'); // Redirect to signup if no session is found
        }
      } catch (error) {
        console.error('Error checking session:', error.message);
        router.push('/signup'); // Handle error by redirecting to signup
      } finally {
        setLoading(false);
      }
    };

    checkSession();
  }, [router]);

  if (loading) {
    return <div>Loading...</div>;
  }

  return null;
}
