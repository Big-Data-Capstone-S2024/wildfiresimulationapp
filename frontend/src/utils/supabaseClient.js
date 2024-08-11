import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Listen for auth state changes
if (typeof window !== "undefined") {
  supabase.auth.onAuthStateChange((event, session) => {
    if (session) {
      localStorage.setItem('supabaseSession', JSON.stringify(session));
    } else {
      localStorage.removeItem('supabaseSession');
    }
  });
}
