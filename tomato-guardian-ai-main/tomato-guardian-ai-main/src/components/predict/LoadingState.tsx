import { Loader2, Cpu, Microscope, Sparkles } from "lucide-react";

const LoadingState = () => {
  return (
    <div className="flex flex-col items-center justify-center py-16 animate-fade-in">
      <div className="relative">
        {/* Outer spinning ring */}
        <div className="w-24 h-24 rounded-full border-4 border-primary/20 border-t-primary animate-spin" />
        
        {/* Inner icon */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
            <Microscope className="h-6 w-6 text-primary animate-pulse" />
          </div>
        </div>
      </div>

      <h3 className="mt-8 text-xl font-semibold text-foreground">Analyzing Image</h3>
      <p className="mt-2 text-muted-foreground text-center max-w-sm">
        Our AI model is processing your tomato leaf image and environmental data to detect potential diseases.
      </p>

      {/* Progress Steps */}
      <div className="mt-8 space-y-3 w-full max-w-xs">
        <LoadingStep icon={<Cpu className="h-4 w-4" />} text="Processing image..." active />
        <LoadingStep icon={<Sparkles className="h-4 w-4" />} text="Running deep learning model..." />
        <LoadingStep icon={<Microscope className="h-4 w-4" />} text="Generating explanations..." />
      </div>
    </div>
  );
};

interface LoadingStepProps {
  icon: React.ReactNode;
  text: string;
  active?: boolean;
}

const LoadingStep = ({ icon, text, active }: LoadingStepProps) => {
  return (
    <div className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
      active ? "bg-primary/10 text-primary" : "text-muted-foreground"
    }`}>
      {active ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : (
        icon
      )}
      <span className="text-sm font-medium">{text}</span>
    </div>
  );
};

export default LoadingState;
