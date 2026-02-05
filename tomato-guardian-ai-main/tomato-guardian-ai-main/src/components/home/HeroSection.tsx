import { ArrowRight, Leaf, Microscope, CloudSun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";

const HeroSection = () => {
  return (
    <section className="relative overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 leaf-pattern opacity-50" />
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-background/50 to-background" />
      
      <div className="container relative py-20 md:py-32">
        <div className="max-w-3xl mx-auto text-center space-y-8 animate-slide-up">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20">
            <Leaf className="h-4 w-4 text-primary" />
            <span className="text-sm font-medium text-primary">AI-Powered Detection</span>
          </div>

          {/* Title */}
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-foreground leading-tight text-balance">
            Tomato Leaf Disease
            <span className="block text-primary">Detection System</span>
          </h1>

          {/* Subtitle */}
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto text-balance">
            Harness the power of multimodal deep learning to detect tomato diseases using leaf images and environmental data. Built for farmers, researchers, and agricultural professionals.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
            <Button variant="hero" size="xl" asChild>
              <Link to="/predict">
                Get Started
                <ArrowRight className="ml-2 h-5 w-5" />
              </Link>
            </Button>
            <Button variant="outline" size="xl" asChild>
              <Link to="/about">Learn More</Link>
            </Button>
          </div>
        </div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-3 gap-6 mt-20 max-w-4xl mx-auto">
          <FeatureCard
            icon={<Microscope className="h-6 w-6" />}
            title="Image Analysis"
            description="Upload tomato leaf photos for instant disease detection using deep learning"
          />
          <FeatureCard
            icon={<CloudSun className="h-6 w-6" />}
            title="Environmental Data"
            description="Combine weather conditions for more accurate predictions"
          />
          <FeatureCard
            icon={<Leaf className="h-6 w-6" />}
            title="Explainable AI"
            description="Visual heatmaps showing which leaf areas influence the diagnosis"
          />
        </div>
      </div>
    </section>
  );
};

interface FeatureCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
}

const FeatureCard = ({ icon, title, description }: FeatureCardProps) => {
  return (
    <div className="glass-card rounded-xl p-6 text-center hover:shadow-elevated transition-all duration-300 hover:-translate-y-1">
      <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-primary/10 text-primary mb-4">
        {icon}
      </div>
      <h3 className="font-semibold text-foreground mb-2">{title}</h3>
      <p className="text-sm text-muted-foreground">{description}</p>
    </div>
  );
};

export default HeroSection;
