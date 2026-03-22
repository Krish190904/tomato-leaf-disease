import { Upload, Cpu, FileCheck } from "lucide-react";

const HowItWorks = () => {
  const steps = [
    {
      icon: <Upload className="h-6 w-6" />,
      step: "01",
      title: "Upload Image",
      description: "Take a clear photo of a tomato leaf and upload it along with optional environmental data.",
    },
    {
      icon: <Cpu className="h-6 w-6" />,
      step: "02",
      title: "AI Analysis",
      description: "Our multimodal deep learning model processes the image and environmental parameters.",
    },
    {
      icon: <FileCheck className="h-6 w-6" />,
      step: "03",
      title: "Get Results",
      description: "Receive instant disease predictions with confidence scores and visual explanations.",
    },
  ];

  return (
    <section className="py-20 bg-muted/30">
      <div className="container">
        <div className="text-center max-w-2xl mx-auto mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
            How It Works
          </h2>
          <p className="text-muted-foreground text-lg">
            Three simple steps to identify tomato plant diseases using our AI-powered system.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          {steps.map((item, index) => (
            <div key={index} className="relative">
              {/* Connector Line */}
              {index < steps.length - 1 && (
                <div className="hidden md:block absolute top-12 left-[60%] w-[80%] h-[2px] bg-gradient-to-r from-primary/50 to-primary/20" />
              )}
              
              <div className="relative bg-card rounded-2xl p-8 shadow-soft hover:shadow-elevated transition-all duration-300 border border-border/50">
                {/* Step Number */}
                <div className="absolute -top-3 -right-3 w-10 h-10 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-bold">
                  {item.step}
                </div>
                
                <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center text-primary mb-6">
                  {item.icon}
                </div>
                
                <h3 className="text-xl font-semibold text-foreground mb-3">{item.title}</h3>
                <p className="text-muted-foreground">{item.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
