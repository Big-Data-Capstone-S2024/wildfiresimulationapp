// src/components/Layout.js
import Header from './Header';
import Footer from './Footer';

export default function Layout({ children }) {
  return (
    <div className="min-h-screen flex flex-col bg-gray-200">
      <Header />
      <main className="flex-grow flex justify-center items-center p-4">
        {children}
      </main>
      <Footer />
    </div>
  );
}
