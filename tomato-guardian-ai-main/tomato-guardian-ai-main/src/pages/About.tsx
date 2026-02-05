import { Leaf, Brain, CloudSun, Target, Users, Microscope } from "lucide-react";
import Layout from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";

const About = () => {
  const features = [
    {
      icon: <Brain className="h-6 w-6" />,
      title: "Multimodal Deep Learning",
      description: "Our system combines convolutional neural networks for image analysis with environmental data processing for comprehensive disease detection.",
    },
    {
      icon: <CloudSun className="h-6 w-6" />,
      title: "Environmental Integration",
      description: "Temperature, humidity, and rainfall data are factored into predictions, as these conditions significantly influence disease development.",
    },
    {
      icon: <Microscope className="h-6 w-6" />,
      title: "Explainable AI",
      description: "Grad-CAM visualizations highlight which parts of the leaf image influenced the model's decision, providing transparency and trust.",
    },
    {
      icon: <Target className="h-6 w-6" />,
      title: "High Accuracy",
      description: "Trained on thousands of real-world field images, our model achieves high accuracy across multiple tomato disease classes.",
    },
  ];

  const diseaseClasses = [
    "Healthy",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Yellow Leaf Curl Virus",
    "Bacterial Spot",
    "Mosaic Virus",
  ];

  return (
    <Layout>
      <section className="py-12 md:py-20">
        <div className="container max-w-5xl">
          {/* Header */}
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 mb-6">
              <Leaf className="h-4 w-4 text-primary" />
              <span className="text-sm font-medium text-primary">About the Project</span>
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              AI-Powered Agriculture
            </h1>
            <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
              Empowering farmers and researchers with cutting-edge deep learning technology for early disease detection and prevention.
            </p>
          </div>

          {/* Mission */}
          <div className="bg-card rounded-2xl border border-border shadow-soft p-8 md:p-12 mb-12">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center shrink-0">
                <Users className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-foreground mb-4">Our Mission</h2>
                <p className="text-muted-foreground leading-relaxed">
                  Tomato cultivation is vital to global food security, but plant diseases can devastate crops and livelihoods. 
                  Traditional disease identification requires expert knowledge that isn't always accessible to small-scale farmers. 
                  Our system bridges this gap by providing instant, accurate disease detection using just a smartphone photo.
                </p>
                <p className="text-muted-foreground leading-relaxed mt-4">
                  By combining visual analysis with environmental data, we provide context-aware predictions that account for 
                  conditions favoring specific diseases. This holistic approach improves accuracy and helps farmers take 
                  preventive action before diseases spread.
                </p>
              </div>
            </div>
          </div>

          {/* Features Grid */}
          <div className="mb-16">
            <h2 className="text-2xl font-bold text-foreground text-center mb-8">Key Features</h2>
            <div className="grid md:grid-cols-2 gap-6">
              {features.map((feature, index) => (
                <div
                  key={index}
                  className="bg-card rounded-xl border border-border p-6 hover:shadow-soft transition-shadow"
                >
                  <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center text-primary mb-4">
                    {feature.icon}
                  </div>
                  <h3 className="text-lg font-semibold text-foreground mb-2">{feature.title}</h3>
                  <p className="text-muted-foreground text-sm">{feature.description}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Supported Diseases */}
          <div className="bg-muted/30 rounded-2xl border border-border/50 p-8 md:p-12 mb-12">
            <h2 className="text-2xl font-bold text-foreground text-center mb-8">
              Supported Disease Classes
            </h2>
            <div className="flex flex-wrap justify-center gap-3">
              {diseaseClasses.map((disease, index) => (
                <span
                  key={index}
                  className={`px-4 py-2 rounded-full text-sm font-medium ${
                    disease === "Healthy"
                      ? "bg-success/10 text-success border border-success/20"
                      : "bg-card text-foreground border border-border"
                  }`}
                >
                  {disease}
                </span>
              ))}
            </div>
          </div>

          {/* Technical Details */}
          <div className="bg-card rounded-2xl border border-border shadow-soft p-8 md:p-12 mb-12">
            <h2 className="text-2xl font-bold text-foreground mb-6">Technical Overview</h2>
            <div className="space-y-4 text-muted-foreground">
              <p>
                <strong className="text-foreground">Model Architecture:</strong> The system uses a multimodal deep learning approach 
                combining a pre-trained CNN (such as ResNet or EfficientNet) for image feature extraction with a fully connected 
                network for environmental data processing. Features are fused and passed through classification layers.
              </p>
              <p>
                <strong className="text-foreground">Training Data:</strong> The model is trained on the PlantVillage dataset 
                augmented with real-world field images, ensuring robustness across various lighting conditions, angles, and 
                disease stages.
              </p>
              <p>
                <strong className="text-foreground">Explainability:</strong> Grad-CAM (Gradient-weighted Class Activation Mapping) 
                generates visual explanations by highlighting image regions that contributed most to the prediction, making the 
                AI's reasoning transparent.
              </p>
              <p>
                <strong className="text-foreground">Backend:</strong> The prediction API is built with FastAPI, providing 
                fast, asynchronous processing with proper validation and error handling. The frontend communicates via 
                RESTful endpoints for seamless integration.
              </p>
            </div>
          </div>

          {/* CTA */}
          <div className="text-center">
            <h2 className="text-2xl font-bold text-foreground mb-4">Ready to Try It?</h2>
            <p className="text-muted-foreground mb-6">
              Upload a tomato leaf image and get instant disease predictions with AI-powered analysis.
            </p>
            <Button size="lg" asChild>
              <Link to="/predict">
                Start Detection
                <Leaf className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </div>
        </div>
      </section>
    </Layout>
  );
};

export default About;
