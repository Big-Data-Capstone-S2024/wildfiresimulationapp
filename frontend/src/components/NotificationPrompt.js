import React, { useEffect } from 'react';
import { useRouter } from 'next/router';

export default function NotificationPrompt() {
  const router = useRouter();

  useEffect(() => {
    const timer = setTimeout(() => {
      router.push('/dashboard');
    }, 3000); // 3-second delay before routing to the dashboard

    return () => clearTimeout(timer);
  }, [router]);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100">
      <img src="/path-to-your-image.png" alt="Notification Prompt" />
    </div>
  );
}
