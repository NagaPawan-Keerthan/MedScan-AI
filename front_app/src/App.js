import React, { useState } from "react";
import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [message, setMessage] = useState("");
  const [reportUrl, setReportUrl] = useState("");

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setMessage("");
    setReportUrl("");
  };

  const handleUpload = async (event) => {
    event.preventDefault();

    if (!selectedFile) {
      setMessage("Please select a file to upload.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:8000/upload/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        setMessage(errorData.message || "Failed to process the file.");
        return;
      }

      const data = await response.json();
      setMessage("Report generated successfully. Click the button to download.");
      setReportUrl(data.report_url);
    } catch (error) {
      console.error("Error uploading file:", error);
      setMessage("An error occurred while uploading the file.");
    }
  };

  const handleDownload = () => {
    if (reportUrl) {
      window.open(reportUrl, "_blank");
    }
  };

  return (
    <div className="App">
      <div className="card">
        <h1>MedScan AI</h1>
        <form onSubmit={handleUpload}>
          <label>
            Upload Query Image:
            <input type="file" onChange={handleFileChange} accept="image/*" />
          </label>
          <button type="submit" className="upload-btn">
            Upload and Process
          </button>
        </form>

        {message && <p>{message}</p>}

        {reportUrl && (
          <button onClick={handleDownload} className="download-btn">
            Download Report
          </button>
        )}
      </div>
    </div>
  );
}

export default App;