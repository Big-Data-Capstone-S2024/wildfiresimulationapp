import { useEffect } from 'react';
import { useRouter } from 'next/router';
import { supabase } from '../utils/supabaseClient';

export default function Confirm() {
  const router = useRouter();

  useEffect(() => {
    const confirmEmail = async () => {
      const { data: { session }, error } = await supabase.auth.getSessionFromUrl();

      if (error || !session) {
        alert('Invalid or expired confirmation link.');
        router.push('/login');
      } else {
        // Save session token to HTTP-only cookie via an API route
        await fetch('/api/storeSession', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ sessionToken: session.access_token }),
        });

        // Redirect to dashboard
        router.push('/dashboard');
      }
    };

    confirmEmail();
  }, [router]);

  return (
    <div className="flex justify-center items-center h-screen">
      <p>Confirming your email...</p>
    </div>
  );
}
