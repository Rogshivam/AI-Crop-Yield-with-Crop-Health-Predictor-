import React, { useEffect, useRef, useState } from 'react'
import axios from 'axios'
import jsPDF from 'jspdf'
import html2canvas from 'html2canvas'

export default function App() {
  // Configure axios base URL from env if provided
  const apiBase = import.meta.env.VITE_API_BASE?.trim()
  if (apiBase) {
    axios.defaults.baseURL = apiBase
  }
  const [streamActive, setStreamActive] = useState(false)
  const [previewUrl, setPreviewUrl] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [serverHealth, setServerHealth] = useState(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const reportRef = useRef(null)

  useEffect(() => {
    let active = true
    const fetchHealth = () => axios.get('/health').then(r => active && setServerHealth(r.data)).catch(() => active && setServerHealth(null))
    fetchHealth()
    const id = setInterval(fetchHealth, 5000)
    return () => { active = false; clearInterval(id) }
  }, [])

  const startCamera = async () => {
    if (streamActive) return
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
        setStreamActive(true)
      }
    } catch (e) {
      setError('Camera access denied or unavailable')
    }
  }

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks()
      tracks.forEach(t => t.stop())
      videoRef.current.srcObject = null
    }
    setStreamActive(false)
  }

  const capturePhoto = () => {
    if (!videoRef.current) return
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!canvas) return
    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    canvas.toBlob(blob => {
      if (!blob) return
      const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' })
      setSelectedFile(file)
      const url = URL.createObjectURL(blob)
      setPreviewUrl(url)
    }, 'image/jpeg', 0.92)
  }

  const onFileChange = e => {
    const file = e.target.files?.[0]
    if (!file) return
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
  }

  const predict = async () => {
    if (!selectedFile) {
      setError('Select or capture an image first')
      return
    }
    setError('')
    setLoading(true)
    setResult(null)
    try {
      const form = new FormData()
      form.append('image', selectedFile)
      const res = await axios.post('/api/predict', form, { headers: { 'Content-Type': 'multipart/form-data' } })
      const now = new Date().toISOString()
      setResult({ ...res.data, timestamp: now })
    } catch (e) {
      const msg = e?.response?.data?.error || 'Prediction failed'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const downloadPdf = async () => {
    if (!reportRef.current) return
    const el = reportRef.current
    const canvas = await html2canvas(el, { scale: 2 })
    const imgData = canvas.toDataURL('image/png')
    const pdf = new jsPDF('p', 'mm', 'a4')
    const pageWidth = pdf.internal.pageSize.getWidth()
    const pageHeight = pdf.internal.pageSize.getHeight()
    const imgWidth = pageWidth - 20
    const imgHeight = canvas.height * imgWidth / canvas.width
    pdf.addImage(imgData, 'PNG', 10, 10, imgWidth, Math.min(imgHeight, pageHeight - 20))
    pdf.save('crop-health-report.pdf')
  }

  const reset = () => {
    setSelectedFile(null)
    setPreviewUrl('')
    setResult(null)
    setError('')
  }

  // (yield flow removed)

  const statusBadge = (ok) => (
    <span style={{ padding: '2px 8px', borderRadius: 999, background: ok ? '#e7f6e7' : '#fde8e8', color: ok ? '#166534' : '#991b1b', fontSize: 12 }}>
      {ok ? 'ready' : 'not ready'}
    </span>
  )

  return (
    <div style={{ fontFamily: 'ui-sans-serif, system-ui, Arial', background: '#f8fafc', minHeight: '100vh' }}>
      <div style={{ maxWidth: 960, margin: '0 auto', padding: 16 }}>
        <h1 style={{ fontSize: 28, fontWeight: 700, margin: '12px 0' }}>AI Crop Health</h1>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12 }}>
          <div>API</div>
          {statusBadge(!!serverHealth)}
          <div>Model</div>
          {statusBadge(!!serverHealth?.health_model_loaded)}
          <div>Labels</div>
          {statusBadge(!!serverHealth?.health_labels_loaded)}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8, padding: 16 }}>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
              <button onClick={startCamera} style={{ background: '#2563eb', color: '#fff', padding: '8px 12px', borderRadius: 6, border: 'none' }}>Start Camera</button>
              <button onClick={capturePhoto} disabled={!streamActive} style={{ background: streamActive ? '#059669' : '#9ca3af', color: '#fff', padding: '8px 12px', borderRadius: 6, border: 'none' }}>Capture</button>
              <button onClick={stopCamera} style={{ background: '#ef4444', color: '#fff', padding: '8px 12px', borderRadius: 6, border: 'none' }}>Stop</button>
              <div style={{ marginLeft: 'auto' }}>
                <input type="file" accept="image/*" onChange={onFileChange} />
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 12 }}>
              <div style={{ background: '#0b1220', borderRadius: 8, minHeight: 240, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                <video ref={videoRef} style={{ width: '100%', height: '100%' }} />
                <canvas ref={canvasRef} style={{ display: 'none' }} />
              </div>
              <div style={{ background: '#f1f5f9', borderRadius: 8, minHeight: 240, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                {previewUrl ? (
                  <img src={previewUrl} alt="preview" style={{ maxWidth: '100%', maxHeight: 320, objectFit: 'contain' }} />
                ) : (
                  <div style={{ color: '#64748b' }}>No image selected</div>
                )}
              </div>
            </div>
            <div style={{ display: 'flex', gap: 12, marginTop: 12 }}>
              <button onClick={predict} disabled={loading || !selectedFile || !serverHealth?.health_model_loaded || !serverHealth?.health_labels_loaded} style={{ background: selectedFile && serverHealth?.health_model_loaded && serverHealth?.health_labels_loaded ? '#0ea5e9' : '#9ca3af', color: '#fff', padding: '8px 12px', borderRadius: 6, border: 'none' }}>{loading ? 'Predicting...' : 'Predict'}</button>
              <button onClick={reset} style={{ background: '#6b7280', color: '#fff', padding: '8px 12px', borderRadius: 6, border: 'none' }}>Reset</button>
            </div>
            {error && <div style={{ color: '#b91c1c', marginTop: 8 }}>{error}</div>}
          </div>

          <div ref={reportRef} style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8, padding: 16 }}>
            <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 8 }}>Report</h2>
            {!result ? (
              <div style={{ color: '#64748b' }}>Run a prediction to see details</div>
            ) : (
              <div>
                <div style={{ display: 'grid', gridTemplateColumns: '160px 1fr', gap: 16 }}>
                  <div style={{ background: '#f1f5f9', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                    {previewUrl && <img src={previewUrl} alt="preview" style={{ width: '100%', objectFit: 'cover' }} />}
                  </div>
                  <div>
                    <div style={{ marginBottom: 6 }}><strong>Label Name:</strong> {result.label_name ?? 'N/A'}</div>
                    <div style={{ marginBottom: 6 }}><strong>Label Index:</strong> {result.label}</div>
                    <div style={{ marginBottom: 6 }}>
                      <strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%
                      <div style={{ height: 8, background: '#e5e7eb', borderRadius: 999, marginTop: 6 }}>
                        <div style={{ width: `${Math.max(0, Math.min(100, result.confidence * 100)).toFixed(1)}%`, height: '100%', background: '#16a34a', borderRadius: 999 }} />
                      </div>
                    </div>
                    <div style={{ marginBottom: 6 }}><strong>Timestamp:</strong> {new Date(result.timestamp).toLocaleString()}</div>
                    <div style={{ marginTop: 10 }}>
                      <div style={{ fontWeight: 700, marginBottom: 4 }}>Notes</div>
                      <ul style={{ paddingLeft: 18, margin: 0 }}>
                        <li>Verify leaf is well-lit and in focus.</li>
                        <li>Capture single leaf centered in frame.</li>
                        <li>If diseased, isolate plant and consult treatment guides.</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: 12 }}>
            <button onClick={downloadPdf} disabled={!result} style={{ background: result ? '#16a34a' : '#9ca3af', color: '#fff', padding: '8px 12px', borderRadius: 6, border: 'none' }}>Download Report (PDF)</button>
          </div>
        </div>

        {/* Yield panel removed as requested */}
      </div>
    </div>
  )
}
