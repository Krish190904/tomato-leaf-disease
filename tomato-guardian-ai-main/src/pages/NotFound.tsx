import { Leaf, Home, ArrowLeft } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import Layout from "@/components/layout/Layout";

const NotFound = () => {
  return (
    <Layout>
      <section className="py-20 md:py-32">
        <div className="container max-w-2xl text-center">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-primary/10 mb-8">
            <Leaf className="h-10 w-10 text-primary" />
          </div>
          
          <h1 className="text-6xl md:text-8xl font-bold text-foreground mb-4">404</h1>
          <h2 className="text-2xl md:text-3xl font-semibold text-foreground mb-4">
            Page Not Found
          </h2>
          <p className="text-muted-foreground text-lg mb-8">
            Oops! The page you're looking for seems to have wilted away. Let's get you back to healthy ground.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" asChild>
              <Link to="/">
                <Home className="h-4 w-4 mr-2" />
                Go Home
              </Link>
            </Button>
            <Button variant="outline" size="lg" asChild>
              <Link to="/predict">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Try Detection
              </Link>
            </Button>
          </div>
        </div>
      </section>
    </Layout>
  );
};

export default NotFound;
