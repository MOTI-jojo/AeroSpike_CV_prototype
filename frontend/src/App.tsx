import { Streamlit, withStreamlitConnection } from "streamlit-component-lib"
import type { ComponentProps } from "streamlit-component-lib"
import React, { useRef, useState, useMemo } from "react"
import { Canvas, useFrame } from "@react-three/fiber"
import { OrbitControls, Line } from "@react-three/drei"
import * as THREE from "three"

const VolleyballCourt3D = ({ trajectory }: { trajectory: any[] }) => {
  const ballRef = useRef<THREE.Mesh>(null)
  
  // Animation state
  const [time, setTime] = useState(0)
  
  // Create CatmullRomCurve3 from trajectory points
  const curve = useMemo(() => {
    if (!trajectory || trajectory.length === 0) return null
    return new THREE.CatmullRomCurve3(
      trajectory.map(p => new THREE.Vector3(p.x, p.y, p.z))
    )
  }, [trajectory])
  
  const totalDuration = trajectory?.length > 0 ? trajectory[trajectory.length - 1].t : 0
  
  useFrame((_, delta) => {
    if (curve && totalDuration > 0) {
      let nextTime = time + delta
      if (nextTime > totalDuration) {
        nextTime = 0 // loop
      }
      setTime(nextTime)
      
      const fraction = nextTime / totalDuration
      const point = curve.getPointAt(fraction)
      if (ballRef.current) {
        ballRef.current.position.copy(point)
      }
    }
  })

  const courtLength = 18.0
  const courtWidth = 9.0
  const netHeight = 2.43
  
  const linePoints = useMemo(() => {
    if (!curve) return []
    return curve.getPoints(200)
  }, [curve])

  return (
    <>
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 20, 10]} castShadow intensity={1} />
      
      {/* Floor */}
      <mesh receiveShadow rotation={[-Math.PI / 2, 0, 0]} position={[9, 0, 0]}>
        <planeGeometry args={[courtLength + 4, courtWidth + 4]} />
        <meshStandardMaterial color="#333333" />
      </mesh>
      
      {/* Court area */}
      <mesh receiveShadow rotation={[-Math.PI / 2, 0, 0]} position={[9, 0.01, 0]}>
        <planeGeometry args={[courtLength, courtWidth]} />
        <meshStandardMaterial color="#e3713b" />
      </mesh>
      
      {/* Court lines */}
      {/* Outline */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[9, 0.02, 0]}>
        <planeGeometry args={[courtLength, courtWidth]} />
        <meshBasicMaterial color="#ffffff" wireframe />
      </mesh>

      {/* Net */}
      <mesh position={[9, netHeight/2, 0]}>
        <boxGeometry args={[0.05, netHeight, courtWidth + 1]} />
        <meshStandardMaterial color="#ffffff" transparent opacity={0.6} />
      </mesh>
      
      {/* Trajectory Line */}
      {linePoints.length > 0 && (
        <Line points={linePoints} color="#E3000F" lineWidth={3} />
      )}
      
      {/* Ball */}
      <mesh ref={ballRef} castShadow>
        <sphereGeometry args={[0.105, 32, 32]} />
        <meshStandardMaterial color="#ffffaa" />
      </mesh>
      
      <OrbitControls target={[9, 0, 0]} />
    </>
  )
}

class App extends React.Component<ComponentProps> {
  componentDidMount() {
    Streamlit.setFrameHeight(600)
  }

  componentDidUpdate() {
    Streamlit.setFrameHeight(600)
  }

  render() {
    const { args, theme } = this.props
    
    const trajectory = []
    if (args.t && args.x && args.y && args.z) {
      for (let i = 0; i < args.t.length; i++) {
        trajectory.push({
          t: args.t[i],
          x: args.x[i],
          y: args.y[i],
          z: args.z[i]
        })
      }
    }

    return (
      <div style={{ width: "100%", height: "600px", background: theme?.backgroundColor || "#1e1e1e", borderRadius: "8px", overflow: "hidden" }}>
        <Canvas shadows camera={{ position: [-2, 3, 0], fov: 60 }}>
          <VolleyballCourt3D trajectory={trajectory} />
        </Canvas>
      </div>
    )
  }
}

export default withStreamlitConnection(App)
