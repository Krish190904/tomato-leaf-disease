import {
  CheckCircle2,
  AlertTriangle,
  Info,
  Leaf,
  TrendingUp,
  Shield,
  Bug,
  Pill,
} from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import type { PredictionResult } from "@/pages/Predict";

interface ResultsDisplayProps {
  result: PredictionResult;
  originalImageUrl: string;
}

const ResultsDisplay = ({ result, originalImageUrl }: ResultsDisplayProps) => {
  const confidencePercentage = result.confidence_pct ?? Math.round(result.confidence * 100);
  const isHealthy = result.prediction === "Healthy";

  const getConfidenceColor = (pct: number) => {
    if (pct >= 80) return "text-success";
    if (pct >= 60) return "text-warning";
    return "text-destructive";
  };

  const getSeverityStyle = (severity: string) => {
    switch (severity) {
      case "Low":
        return "bg-green-100 text-green-800 border-green-200";
      case "Medium":
        return "bg-amber-100 text-amber-800 border-amber-200";
      case "High":
        return "bg-red-100 text-red-800 border-red-200";
      default:
        return "bg-muted text-muted-foreground border-border";
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Result Header */}
      <div
        className={cn(
          "rounded-2xl p-6 border-2",
          isHealthy
            ? "bg-success/10 border-success/30"
            : "bg-tomato/10 border-tomato/30"
        )}
      >
        <div className="flex items-start gap-4">
          <div
            className={cn(
              "w-12 h-12 rounded-full flex items-center justify-center shrink-0 text-2xl",
              isHealthy ? "bg-success/20" : "bg-tomato/20"
            )}
          >
            {result.emoji || (isHealthy ? (
              <CheckCircle2 className="h-6 w-6 text-success" />
            ) : (
              <AlertTriangle className="h-6 w-6 text-tomato" />
            ))}
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-1">
              <h3 className="text-xl font-bold text-foreground">
                {result.prediction}
              </h3>
              {result.severity && (
                <span
                  className={cn(
                    "text-xs font-semibold px-2.5 py-1 rounded-full border",
                    getSeverityStyle(result.severity)
                  )}
                >
                  {result.severity} Severity
                </span>
              )}
            </div>
            <p className="text-muted-foreground">
              {isHealthy
                ? "Your tomato plant appears to be healthy with no visible signs of disease."
                : result.description || "Disease detected. Consult an agricultural expert for treatment."}
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
          <span
            className={cn(
              "text-2xl font-bold",
              getConfidenceColor(confidencePercentage)
            )}
          >
            {confidencePercentage}%
          </span>
        </div>
        <Progress value={confidencePercentage} className="h-3" />
        <div className="flex justify-between items-center mt-3">
          <p className="text-sm text-muted-foreground">
            {confidencePercentage >= 80
              ? "High confidence prediction"
              : confidencePercentage >= 60
              ? "Moderate confidence — consider additional verification"
              : "Low confidence — manual inspection recommended"}
          </p>
          {result.model_used && (
            <span className="text-xs text-muted-foreground bg-muted px-2 py-1 rounded">
              {result.model_used}
            </span>
          )}
        </div>
      </div>

      {/* Image Preview */}
      <div className="bg-card rounded-xl p-4 border border-border shadow-soft">
        <div className="flex items-center gap-2 mb-3">
          <Leaf className="h-4 w-4 text-primary" />
          <h4 className="font-medium text-foreground">Uploaded Image</h4>
        </div>
        <div className="rounded-lg overflow-hidden bg-muted">
          <img
            src={originalImageUrl}
            alt="Original tomato leaf"
            className="w-full h-56 object-contain"
          />
        </div>
      </div>

      {/* Cause */}
      {result.cause && !isHealthy && (
        <div className="bg-card rounded-xl p-6 border border-border shadow-soft">
          <div className="flex items-start gap-3">
            <Bug className="h-5 w-5 text-tomato mt-0.5 shrink-0" />
            <div>
              <h4 className="font-semibold text-foreground mb-2">What Causes This?</h4>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {result.cause}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Treatment */}
      {result.treatment && result.treatment.length > 0 && !isHealthy && (
        <div className="bg-card rounded-xl p-6 border border-border shadow-soft">
          <div className="flex items-center gap-2 mb-4">
            <Pill className="h-5 w-5 text-primary" />
            <h4 className="font-semibold text-foreground">Treatment Protocol</h4>
          </div>
          <ul className="space-y-3">
            {result.treatment.map((step, i) => (
              <li
                key={i}
                className="flex items-start gap-3 text-sm text-muted-foreground"
              >
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
                  {i + 1}
                </span>
                <span className="leading-relaxed pt-0.5">{step}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Disclaimer */}
      <div className="bg-muted/50 rounded-xl p-6 border border-border/50">
        <div className="flex items-start gap-3">
          <Shield className="h-5 w-5 text-primary mt-0.5 shrink-0" />
          <div>
            <h4 className="font-medium text-foreground mb-2">Disclaimer</h4>
            <p className="text-sm text-muted-foreground">
              This prediction is generated by an AI model and should be used as a preliminary assessment only.
              Always consult with an agricultural expert before applying any treatments to your crops.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;
