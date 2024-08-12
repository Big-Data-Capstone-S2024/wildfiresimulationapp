import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Listen for auth state changes
supabase.auth.onAuthStateChange((event, session) => {
  // Here you can trigger an API call to store the session in a cookie or 
  // manage the session state through another secure mechanism.
  if (event === 'SIGNED_IN') {
    // You might want to save the session token securely here, e.g., through an API route
    fetch('/api/storeSession', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ sessionToken: session.access_token }),
    });
  } else if (event === 'SIGNED_OUT') {
    // Clear the session
    fetch('/api/clearSession', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }
});
