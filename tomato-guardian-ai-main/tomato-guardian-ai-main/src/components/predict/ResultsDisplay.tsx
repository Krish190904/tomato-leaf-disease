import { CheckCircle2, AlertTriangle, Info, Leaf, TrendingUp } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface PredictionResult {
  disease: string;
  confidence: number;
  gradcamUrl?: string;
  isHealthy: boolean;
}

interface ResultsDisplayProps {
  result: PredictionResult;
  originalImageUrl: string;
}

const ResultsDisplay = ({ result, originalImageUrl }: ResultsDisplayProps) => {
  const confidencePercentage = Math.round(result.confidence * 100);
  
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "text-success";
    if (confidence >= 0.6) return "text-warning";
    return "text-destructive";
  };

  const getConfidenceBg = (confidence: number) => {
    if (confidence >= 0.8) return "bg-success";
    if (confidence >= 0.6) return "bg-warning";
    return "bg-destructive";
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Result Header */}
      <div className={cn(
        "rounded-2xl p-6 border-2",
        result.isHealthy 
          ? "bg-success/10 border-success/30" 
          : "bg-tomato/10 border-tomato/30"
      )}>
        <div className="flex items-start gap-4">
          <div className={cn(
            "w-12 h-12 rounded-full flex items-center justify-center shrink-0",
            result.isHealthy ? "bg-success/20" : "bg-tomato/20"
          )}>
            {result.isHealthy ? (
              <CheckCircle2 className="h-6 w-6 text-success" />
            ) : (
              <AlertTriangle className="h-6 w-6 text-tomato" />
            )}
          </div>
          <div className="flex-1">
            <h3 className="text-xl font-bold text-foreground mb-1">
              {result.disease}
            </h3>
            <p className="text-muted-foreground">
              {result.isHealthy 
                ? "Your tomato plant appears to be healthy with no visible signs of disease."
                : "Disease detected. Consider consulting with an agricultural expert for treatment options."}
            </p>
          </div>
        </div>
      </div>

      {/* Confidence Score */}
      <div className="bg-card rounded-xl p-6 border border-border shadow-soft">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-primary" />
            <h4 className="font-semibold text-foreground">Confidence Score</h4>
          </div>
          <span className={cn("text-2xl font-bold", getConfidenceColor(result.confidence))}>
            {confidencePercentage}%
          </span>
        </div>
        <Progress 
          value={confidencePercentage} 
          className="h-3"
          // The progress indicator color is handled by the component
        />
        <p className="text-sm text-muted-foreground mt-3">
          {confidencePercentage >= 80 
            ? "High confidence prediction" 
            : confidencePercentage >= 60 
            ? "Moderate confidence - consider additional verification"
            : "Low confidence - manual inspection recommended"}
        </p>
      </div>

      {/* Image Comparison */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Original Image */}
        <div className="bg-card rounded-xl p-4 border border-border shadow-soft">
          <div className="flex items-center gap-2 mb-3">
            <Leaf className="h-4 w-4 text-primary" />
            <h4 className="font-medium text-foreground">Original Image</h4>
          </div>
          <div className="rounded-lg overflow-hidden bg-muted">
            <img
              src={originalImageUrl}
              alt="Original tomato leaf"
              className="w-full h-48 object-contain"
            />
          </div>
        </div>

        {/* Grad-CAM Visualization */}
        <div className="bg-card rounded-xl p-4 border border-border shadow-soft">
          <div className="flex items-center gap-2 mb-3">
            <Info className="h-4 w-4 text-accent" />
            <h4 className="font-medium text-foreground">AI Attention Map</h4>
          </div>
          <div className="rounded-lg overflow-hidden bg-muted relative">
            {result.gradcamUrl ? (
              <img
                src={result.gradcamUrl}
                alt="Grad-CAM heatmap overlay"
                className="w-full h-48 object-contain"
              />
            ) : (
              <div className="w-full h-48 flex items-center justify-center">
                <p className="text-sm text-muted-foreground">Heatmap visualization</p>
              </div>
            )}
            {/* Simulated heatmap overlay for demo */}
            <div className="absolute inset-0 bg-gradient-to-br from-tomato/30 via-warning/20 to-transparent rounded-lg pointer-events-none" />
          </div>
        </div>
      </div>

      {/* Explanation */}
      <div className="bg-muted/50 rounded-xl p-6 border border-border/50">
        <div className="flex items-start gap-3">
          <Info className="h-5 w-5 text-primary mt-0.5 shrink-0" />
          <div>
            <h4 className="font-medium text-foreground mb-2">Understanding the Visualization</h4>
            <p className="text-sm text-muted-foreground">
              The highlighted regions in the attention map show areas of the leaf that most influenced the model's prediction. 
              Warmer colors (red/orange) indicate higher importance, while cooler colors indicate lower focus. 
              This helps understand which visual features the AI used for disease classification.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;
