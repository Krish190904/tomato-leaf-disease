import { Leaf } from "lucide-react";
import { Link } from "react-router-dom";

const Footer = () => {
  return (
    <footer className="border-t border-border/50 bg-muted/30">
      <div className="container py-8 md:py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Brand */}
          <div className="space-y-4">
            <Link to="/" className="flex items-center gap-2">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary">
                <Leaf className="h-5 w-5 text-primary-foreground" />
              </div>
              <span className="font-semibold text-foreground">FarmGuard AI</span>
            </Link>
            <p className="text-sm text-muted-foreground max-w-xs">
              AI-powered tomato leaf disease detection for sustainable agriculture and improved crop health.
            </p>
          </div>

          {/* Links */}
          <div className="space-y-4">
            <h4 className="font-medium text-foreground">Quick Links</h4>
            <nav className="flex flex-col gap-2">
              <Link to="/" className="text-sm text-muted-foreground hover:text-primary transition-colors">
                Home
              </Link>
              <Link to="/predict" className="text-sm text-muted-foreground hover:text-primary transition-colors">
                Detect Disease
              </Link>
              <Link to="/about" className="text-sm text-muted-foreground hover:text-primary transition-colors">
                About
              </Link>
            </nav>
          </div>

          {/* Disclaimer */}
          <div className="space-y-4">
            <h4 className="font-medium text-foreground">Disclaimer</h4>
            <p className="text-sm text-muted-foreground">
              This project is developed for academic and research purposes. Predictions should be verified by agricultural experts before taking action.
            </p>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t border-border/50 flex flex-col sm:flex-row justify-between items-center gap-4">
          <p className="text-sm text-muted-foreground">
            © {new Date().getFullYear()} FarmGuard AI — Tomato Disease Detection
          </p>
          <p className="text-sm text-muted-foreground">
            Final Year Engineering Project
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
