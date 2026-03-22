import { useState } from "react";
import { Leaf, Send, RotateCcw } from "lucide-react";
import Layout from "@/components/layout/Layout";
import ImageUpload from "@/components/predict/ImageUpload";
import EnvironmentalInputs from "@/components/predict/EnvironmentalInputs";
import ResultsDisplay from "@/components/predict/ResultsDisplay";
import LoadingState from "@/components/predict/LoadingState";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

const API_URL = "http://localhost:8000";

interface EnvironmentalData {
  temperature: string;
  humidity: string;
  rainfall: string;
}

export interface PredictionResult {
  prediction: string;
  confidence: number;
  confidence_pct: number;
  severity: string;
  emoji: string;
  description: string;
  cause: string;
  treatment: string[];
  model_used: string;
}

const Predict = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [environmentalData, setEnvironmentalData] = useState<EnvironmentalData>({
    temperature: "",
    humidity: "",
    rainfall: "",
  });
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");

  const handleImageSelect = (file: File | null) => {
    setSelectedImage(file);
    if (file) {
      setPreviewUrl(URL.createObjectURL(file));
    } else {
      setPreviewUrl("");
    }
    setResult(null);
  };

  const getRainfallCategory = (mm: string): string => {
    const val = parseFloat(mm);
    if (isNaN(val) || val <= 10) return "low";
    if (val <= 30) return "moderate";
    return "high";
  };

  const handleSubmit = async () => {
    if (!selectedImage) {
      toast.error("Please upload a tomato leaf image");
      return;
    }

    setIsLoading(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedImage);
      formData.append("temperature", environmentalData.temperature || "25");
      formData.append("humidity", environmentalData.humidity || "70");
      formData.append("rainfall", getRainfallCategory(environmentalData.rainfall));

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed");
      }

      setResult(data);
      toast.success("Analysis complete!");
    } catch (error: any) {
      const message =
        error.message === "Failed to fetch"
          ? "Cannot reach the backend server. Make sure it's running on http://localhost:8000"
          : error.message || "Failed to analyze image. Please try again.";
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setPreviewUrl("");
    setEnvironmentalData({ temperature: "", humidity: "", rainfall: "" });
    setResult(null);
  };

  return (
    <Layout>
      <section className="py-12 md:py-20">
        <div className="container max-w-4xl">
          {/* Header */}
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 mb-6">
              <Leaf className="h-4 w-4 text-primary" />
              <span className="text-sm font-medium text-primary">Disease Detection</span>
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              Analyze Your Tomato Leaf
            </h1>
            <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
              Upload a clear image of a tomato leaf and optionally provide environmental conditions for more accurate predictions.
            </p>
          </div>

          {!result && !isLoading && (
            <div className="bg-card rounded-2xl border border-border shadow-soft p-6 md:p-8 space-y-8">
              {/* Image Upload */}
              <ImageUpload
                onImageSelect={handleImageSelect}
                selectedImage={selectedImage}
              />

              {/* Divider */}
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-border" />
                </div>
                <div className="relative flex justify-center">
                  <span className="bg-card px-4 text-sm text-muted-foreground">
                    Additional Information
                  </span>
                </div>
              </div>

              {/* Environmental Inputs */}
              <EnvironmentalInputs
                data={environmentalData}
                onChange={setEnvironmentalData}
              />

              {/* Submit Button */}
              <div className="flex justify-center pt-4">
                <Button
                  size="lg"
                  onClick={handleSubmit}
                  disabled={!selectedImage}
                  className="min-w-[200px]"
                >
                  <Send className="h-4 w-4 mr-2" />
                  Detect Disease
                </Button>
              </div>
            </div>
          )}

          {/* Loading State */}
          {isLoading && (
            <div className="bg-card rounded-2xl border border-border shadow-soft p-6 md:p-8">
              <LoadingState />
            </div>
          )}

          {/* Results */}
          {result && !isLoading && (
            <div className="space-y-6">
              <ResultsDisplay result={result} originalImageUrl={previewUrl} />

              <div className="flex justify-center">
                <Button variant="outline" size="lg" onClick={handleReset}>
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Analyze Another Image
                </Button>
              </div>
            </div>
          )}
        </div>
      </section>
    </Layout>
  );
};

export default Predict;
