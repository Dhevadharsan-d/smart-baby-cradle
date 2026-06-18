import { useState, useEffect } from 'react';
import { onAuthStateChanged, User } from 'firebase/auth';
import { auth } from './lib/firebase';



import Navigation from './components/Navigation';
import Dashboard from './components/Dashboard';
import History from './components/History';
import Settings from './components/Settings';
import Login from './components/Login';
import Signup from './components/Signup';
import Profile from './components/Profile'; // <-- Added Profile import

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoginView, setIsLoginView] = useState(true);

  // Listen to Firebase Auth state
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setIsLoading(false);
    });
    return () => unsubscribe();
  }, []);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-lg text-gray-600 animate-pulse">Loading app...</div>
      </div>
    );
  }

  // Show Auth screens if not logged in
  if (!user) {
    return isLoginView ? (
      <Login onSwitchView={() => setIsLoginView(false)} />
    ) : (
      <Signup onSwitchView={() => setIsLoginView(true)} />
    );
  }

  // Main App rendering
  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard />;
      case 'history':
        return <History />;
      case 'profile':          // <-- Added Profile case
        return <Profile />;
      case 'settings':
        return <Settings />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <Navigation currentPage={currentPage} onNavigate={setCurrentPage} />
      {renderPage()}
    </div>
  );
}

export default App;