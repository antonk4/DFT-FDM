#print(f'Start window initialization at {time.perf_counter() - startTime:.2f} seconds')
#[h*(np.mod(i, N))-h/2*N, h*(np.floor(np.mod(i, N**2)/N))-h/2*N, h*(np.floor(i/N**2))-h/2*N, np.sign(eigenstate[i])]

from pygame import display, event, key, draw, locals, quit
import pygame.time

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from scipy import sparse
import os
import time

startTime = time.perf_counter()

h = 0.05
N = 100
eigen = 1
eigens = 5
totalProb = 0.95

calculate = True
render = True

SCREEN_SIZE = (800, 600)
WINDOW_CREATION_FLAGS = locals.OPENGL | locals.DOUBLEBUF
FRAMERATE = 60

class Fishnet:
  @staticmethod
  def setUpOpenGL():
    global FishnetShader, viewMatrix_location_Fishnet
    global projectionMatrix
    FishnetShader = extraFunctions.make_shader('shaders/FishnetVertex.txt', 'shaders/FishnetFragment.txt')

    glUseProgram(FishnetShader)
    projectionMatrix_location_Fishnet = glGetUniformLocation(FishnetShader, 'projectionMatrix')
    glUniformMatrix4fv(projectionMatrix_location_Fishnet, 1, GL_TRUE, projectionMatrix)
    
    viewMatrix_location_Fishnet = glGetUniformLocation(FishnetShader, 'viewMatrix')
  def updateOpenGL(self):
    global FishnetShader, pointVAO, vertexBuffer, pointVecs
    glUseProgram(FishnetShader)
    glBindVertexArray(pointVAO)
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer)
    glBufferData(GL_ARRAY_BUFFER, pointVecs.nbytes, pointVecs, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
  def calcPoints(self):
    global eigenstate_3d, pointVecs
    mask_threshold = (eigenstate_3d**2 / h**3) > threshold

    has_all_neighbors = (
      extraFunctions.get_shifted_mask(mask_threshold, 1, 0, 0) & extraFunctions.get_shifted_mask(mask_threshold, -1, 0, 0) &
      extraFunctions.get_shifted_mask(mask_threshold, 0, 1, 0) & extraFunctions.get_shifted_mask(mask_threshold, 0, -1, 0) &
      extraFunctions.get_shifted_mask(mask_threshold, 0, 0, 1) & extraFunctions.get_shifted_mask(mask_threshold, 0, 0, -1)
    )
    zs, ys, xs = tuple(h*i-h/2*N for i in np.where(mask_threshold & ~has_all_neighbors))
    signs = np.sign(eigenstate_3d[mask_threshold & ~has_all_neighbors])
    pointVecs = np.column_stack((xs, ys, zs, signs)).astype(np.float32)
  def draw(self):
    glUseProgram(FishnetShader)
    glUniformMatrix4fv(viewMatrix_location_Fishnet, 1, GL_TRUE, viewMatrix)
    glBindVertexArray(pointVAO)
    glDrawArrays(GL_POINTS, 0, len(pointVecs))
  def changeEigen(self, change):
    global eigen, FishnetShader, pointVAO, vertexBuffer, pointVecs, eigenvalues
    setUpFunctions.setUpPoints(h, N, eigen+change)
    eigen += change
    glUseProgram(FishnetShader)
    glBindVertexArray(pointVAO)
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer)
    glBufferData(GL_ARRAY_BUFFER, pointVecs.nbytes, pointVecs, GL_STATIC_DRAW)
    display.set_caption(f'Waveform Render of Eigenvalue {eigen} at {eigenvalues[eigen-1]}')
  def changeRenderType(self):
    global renderType
    renderType = SolidFlat()

    print(f'Start calcpoints at {time.perf_counter() - startTime:.2f} seconds')
    renderType.calcPoints()
    print(f'End calcpoints at {time.perf_counter() - startTime:.2f} seconds')
    renderType.updateOpenGL()
  
class SolidFlat:
  @staticmethod
  def setUpOpenGL():
    global SolidFlatShader, viewMatrix_location_SolidFlat
    global projectionMatrix
    SolidFlatShader = extraFunctions.make_shader('shaders/SolidFlatVertex.txt', 'shaders/SolidFlatFragment.txt')

    glUseProgram(SolidFlatShader)
    projectionMatrix_location_SolidFlat = glGetUniformLocation(SolidFlatShader, 'projectionMatrix')
    glUniformMatrix4fv(projectionMatrix_location_SolidFlat, 1, GL_TRUE, projectionMatrix)
    
    viewMatrix_location_SolidFlat = glGetUniformLocation(SolidFlatShader, 'viewMatrix')
  def updateOpenGL(self):
    global SolidFlatShader, pointVAO, vertexBuffer, pointVecs, indicesBuffer, triangles
    glUseProgram(SolidFlatShader)
    glBindVertexArray(pointVAO)
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer)
    glBufferData(GL_ARRAY_BUFFER, pointVecs.nbytes, pointVecs, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indiceBuffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles.nbytes, triangles, GL_STATIC_DRAW)
  def calcPoints(self):
    global eigenstate_3d, triangles, pointVecs
    filled = (eigenstate_3d[1:-1, 1:-1, 1:-1]**2)/h**3 >= threshold

    bottom   = (eigenstate_3d[0:-2, 1:-1, 1:-1]**2)/h**3 < threshold
    top  = (eigenstate_3d[2:,   1:-1, 1:-1]**2)/h**3 < threshold
    right = (eigenstate_3d[1:-1, 0:-2, 1:-1]**2)/h**3 < threshold
    left    = (eigenstate_3d[1:-1, 2:,   1:-1]**2)/h**3 < threshold
    back   = (eigenstate_3d[1:-1, 1:-1, 0:-2]**2)/h**3 < threshold
    front  = (eigenstate_3d[1:-1, 1:-1, 2:]**2)/h**3   < threshold

    zs, ys, xs = np.where(filled)
    signs = np.sign(eigenstate_3d[np.where(filled)])

    face_offsets = {
      'bottom': (bottom, [[-0.5,-0.5,-0.5,0], [-0.5, 0.5,-0.5,0], [0.5,-0.5, -0.5,0], [0.5,-0.5, -0.5,0], [-0.5, 0.5,-0.5,0], [0.5, 0.5, -0.5,0]]),
      'top': (top, [[-0.5,-0.5,0.5,0], [0.5,-0.5, 0.5,0], [-0.5, 0.5,0.5,0], [0.5,-0.5, 0.5,0], [0.5, 0.5, 0.5,0], [-0.5, 0.5,0.5,0]]),
      'right': (right, [[-0.5,-0.5,-0.5,0], [ 0.5,-0.5,-0.5,0], [-0.5,-0.5, 0.5,0], [-0.5,-0.5, 0.5,0], [ 0.5,-0.5,-0.5,0], [ 0.5,-0.5, 0.5,0]]),
      'left': (left, [[-0.5, 0.5,-0.5,0], [-0.5, 0.5, 0.5,0], [ 0.5, 0.5,-0.5,0], [-0.5, 0.5, 0.5,0], [ 0.5, 0.5, 0.5,0], [ 0.5, 0.5,-0.5,0]]),
      'back': (back, [[-0.5,-0.5,-0.5,0], [-0.5,-0.5,0.5,0], [-0.5, 0.5,-0.5,0], [-0.5, 0.5,-0.5,0], [-0.5,-0.5,0.5,0], [-0.5,0.5,0.5,0]]),
      'front': (front, [[0.5,-0.5,-0.5,0], [0.5,0.5,-0.5,0], [0.5,-0.5,0.5,0], [0.5,0.5,-0.5,0], [0.5, 0.5, 0.5,0], [0.5,-0.5, 0.5,0]])
    }

    all_vertices = []

    for faces, (neighbor_mask, offsets) in face_offsets.items():
      exposed_mask = neighbor_mask[zs, ys, xs]

      cx, cy, cz = xs[exposed_mask], ys[exposed_mask], zs[exposed_mask]
      csigns = signs[exposed_mask]
      
      centers = np.stack([cx+1, cy+1, cz+1, csigns], axis=-1)[:, np.newaxis, :]
      
      face_verts = centers + np.array(offsets)
      all_vertices.append(face_verts.reshape(-1, 4))

    raw_points = np.vstack(all_vertices).astype(np.float32)
    
    scale_factors = np.array([h, h, h], dtype=np.float32)
    shift_factors = np.array([0.5 * h * N, 0.5 * h * N, 0.5 * h * N], dtype=np.float32)
    keys = raw_points[:, :3] * scale_factors - shift_factors
    keys = np.round(keys, decimals=6)
    
    _, unique_indices, inverse_indices = np.unique(
      keys, axis=0, return_index=True, return_inverse=True
    )
    
    pointVecs = raw_points[unique_indices]
    pointVecs[:, :3] = ((pointVecs[:, :3] * h) - (0.5 * h * N))
    triangles = inverse_indices.astype(np.uint32)
  def draw(self):
    glUseProgram(SolidFlatShader)
    glUniformMatrix4fv(viewMatrix_location_SolidFlat, 1, GL_TRUE, viewMatrix)
    glBindVertexArray(pointVAO)
    glDrawElements(GL_TRIANGLES, len(triangles), GL_UNSIGNED_INT, ctypes.c_void_p(0))
  def changeEigen(self, change):
    global eigen, SolidFlatShader, pointVAO, vertexBuffer, pointVecs, indiceBuffer, triangles, eigenvalues
    setUpFunctions.setUpPoints(h, N, eigen+change)
    eigen += change
    glUseProgram(SolidFlatShader)
    glBindVertexArray(pointVAO)
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer)
    glBufferData(GL_ARRAY_BUFFER, pointVecs.nbytes, pointVecs, GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indiceBuffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles.nbytes, triangles, GL_STATIC_DRAW)
    display.set_caption(f'Waveform Render of Eigenvalue {eigen} at {eigenvalues[eigen-1]}')
  def changeRenderType(self):
    global renderType
    renderType = SolidShaded()

    print(f'Start calcpoints at {time.perf_counter() - startTime:.2f} seconds')
    renderType.calcPoints()
    print(f'End calcpoints at {time.perf_counter() - startTime:.2f} seconds')
    renderType.updateOpenGL()
        
class SolidShaded:
  @staticmethod
  def setUpOpenGL():
    global SolidShadedShader, viewMatrix_location_SolidShaded
    global projectionMatrix
    SolidShadedShader = extraFunctions.make_shader('shaders/SolidShadedVertex.txt', 'shaders/SolidShadedFragment.txt')

    glUseProgram(SolidShadedShader)
    projectionMatrix_location_SolidShaded = glGetUniformLocation(SolidShadedShader, 'projectionMatrix')
    glUniformMatrix4fv(projectionMatrix_location_SolidShaded, 1, GL_TRUE, projectionMatrix)
    
    viewMatrix_location_SolidShaded = glGetUniformLocation(SolidShadedShader, 'viewMatrix')
  def updateOpenGL(self):
    global SolidShadedShader, pointVAO, vertexBuffer, pointVecs, occlusionBuffer, occlusion, indicesBuffer, triangles
    glUseProgram(SolidShadedShader)
    glBindVertexArray(pointVAO)
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer)
    glBufferData(GL_ARRAY_BUFFER, pointVecs.nbytes, pointVecs, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))

    glBindBuffer(GL_ARRAY_BUFFER, occlusionBuffer)
    glBufferData(GL_ARRAY_BUFFER, occlusion.nbytes, occlusion, GL_STATIC_DRAW)
    glEnableVertexAttribArray(1)
    glVertexAttribIPointer(1, 1, GL_UNSIGNED_INT, np.uint32(0).nbytes, ctypes.c_void_p(0))

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indiceBuffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles.nbytes, triangles, GL_STATIC_DRAW)
  def calcPoints(self):
    global eigenstate_3d, triangles, pointVecs, occlusion
    filled = (eigenstate_3d[1:-1, 1:-1, 1:-1]**2)/h**3 >= threshold

    bottom = (eigenstate_3d[0:-2, 1:-1, 1:-1]**2)/h**3 < threshold
    top = (eigenstate_3d[2:, 1:-1, 1:-1]**2)/h**3 < threshold
    right = (eigenstate_3d[1:-1, 0:-2, 1:-1]**2)/h**3 < threshold
    left = (eigenstate_3d[1:-1, 2:, 1:-1]**2)/h**3 < threshold
    back = (eigenstate_3d[1:-1, 1:-1, 0:-2]**2)/h**3 < threshold
    front = (eigenstate_3d[1:-1, 1:-1, 2:]**2)/h**3 < threshold

    zs, ys, xs = np.where(filled)
    signs = np.sign(eigenstate_3d[np.where(filled)])

    face_offsets = {
      'bottom': (bottom, [[-0.5,-0.5,-0.5,0], [-0.5,0.5,-0.5,0], [0.5,-0.5,-0.5,0], [0.5,-0.5,-0.5,0], [-0.5,0.5,-0.5,0], [0.5,0.5,-0.5,0]]),
      'top': (top, [[-0.5,-0.5,0.5,0], [0.5,-0.5,0.5,0], [-0.5,0.5,0.5,0], [0.5,-0.5,0.5,0], [0.5,0.5,0.5,0], [-0.5,0.5,0.5,0]]),
      'right': (right, [[-0.5,-0.5,-0.5,0], [0.5,-0.5,-0.5,0], [-0.5,-0.5,0.5,0], [-0.5,-0.5,0.5,0], [0.5,-0.5,-0.5,0], [0.5,-0.5,0.5,0]]),
      'left': (left, [[-0.5,0.5,-0.5,0], [-0.5,0.5,0.5,0], [0.5,0.5,-0.5,0], [-0.5,0.5,0.5,0], [0.5,0.5,0.5,0], [0.5,0.5,-0.5,0]]),
      'back': (back, [[-0.5,-0.5,-0.5,0], [-0.5,-0.5,0.5,0], [-0.5,0.5,-0.5,0], [-0.5,0.5,-0.5,0], [-0.5,-0.5,0.5,0], [-0.5,0.5,0.5,0]]),
      'front': (front, [[0.5,-0.5,-0.5,0], [0.5,0.5,-0.5,0], [0.5,-0.5,0.5,0], [0.5,0.5,-0.5,0], [0.5,0.5,0.5,0], [0.5,-0.5,0.5,0]])
    }

    all_vertices = []

    for faces, (neighbor_mask, offsets) in face_offsets.items():
      exposed_mask = neighbor_mask[zs, ys, xs]

      cx, cy, cz = xs[exposed_mask], ys[exposed_mask], zs[exposed_mask]
      csigns = signs[exposed_mask]
      
      centers = np.stack([cx+1, cy+1, cz+1, csigns], axis=-1)[:, np.newaxis, :]
      
      face_verts = centers + np.array(offsets)
      all_vertices.append(face_verts.reshape(-1, 4))

    raw_points = np.vstack(all_vertices).astype(np.float32)
    
    scale_factors = np.array([h, h, h], dtype=np.float32)
    shift_factors = np.array([0.5 * h * N, 0.5 * h * N, 0.5 * h * N], dtype=np.float32)
    keys = raw_points[:, :3] * scale_factors - shift_factors
    keys = np.round(keys, decimals=6)
    
    _, unique_indices, inverse_indices = np.unique(
      keys, axis=0, return_index=True, return_inverse=True
    )
    
    pointVecs = raw_points[unique_indices]

    pointVecs[:, :3] = ((pointVecs[:, :3] * h) - (0.5 * h * N))
    triangles = inverse_indices.astype(np.uint32)

    nnn = (eigenstate_3d[0:-1, 0:-1, 0:-1]**2)/h**3 < threshold
    pnn = (eigenstate_3d[1:, 0:-1, 0:-1]**2)/h**3 < threshold
    npn = (eigenstate_3d[0:-1, 1:, 0:-1]**2)/h**3 < threshold
    nnp = (eigenstate_3d[0:-1, 0:-1, 1:]**2)/h**3 < threshold
    ppn = (eigenstate_3d[1:, 1:, 0:-1]**2)/h**3 < threshold
    pnp = (eigenstate_3d[1:, 0:-1, 1:]**2)/h**3 < threshold
    npp = (eigenstate_3d[0:-1, 1:, 1:]**2)/h**3 < threshold
    ppp = (eigenstate_3d[1:, 1:, 1:]**2)/h**3 < threshold

    diagonal_neighbors = np.clip(np.sum(np.array([nnn, pnn, npn, nnp, ppn, pnp, npp, ppp]), axis=0), 1, 4)

    step1_points = raw_points[unique_indices]
    step2_points = (step1_points[:,:3] - 0.5).astype(np.uint32)
    column1 = step2_points[:, 0]
    column2 = step2_points[:, 1]
    column3 = step2_points[:, 2]
    
    occlusion = diagonal_neighbors[column3, column2, column1].astype(np.uint32)
  def draw(self):
    glUseProgram(SolidShadedShader)
    glUniformMatrix4fv(viewMatrix_location_SolidShaded, 1, GL_TRUE, viewMatrix)
    glBindVertexArray(pointVAO)
    glDrawElements(GL_TRIANGLES, len(triangles), GL_UNSIGNED_INT, ctypes.c_void_p(0))
  def changeEigen(self, change):
    global eigen, SolidShadedShader, pointvAO, verte, pointVecs, occlusionBuffer, occlusion, indiceBuffer, triangles, eigenvalues
    setUpFunctions.setUpPoints(h, N, eigen+change)
    eigen += change
    glUseProgram(SolidShadedShader)
    glBindVertexArray(pointVAO)
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer)
    glBufferData(GL_ARRAY_BUFFER, pointVecs.nbytes, pointVecs, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, occlusionBuffer)
    glBufferData(GL_ARRAY_BUFFER, occlusion.nbytes, occlusion, GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indiceBuffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles.nbytes, triangles, GL_STATIC_DRAW)
    display.set_caption(f'Waveform Render of Eigenvalue {eigen} at {eigenvalues[eigen-1]}')
  def changeRenderType(self):
    global renderType
    renderType = SolidSmooth()

    print(f'Start calcpoints at {time.perf_counter() - startTime:.2f} seconds')
    renderType.calcPoints()
    print(f'End calcpoints at {time.perf_counter() - startTime:.2f} seconds')
    renderType.updateOpenGL()

class SolidSmooth:
  @staticmethod
  def setUpOpenGL():
    global SolidSmoothShader, viewMatrix_location_SolidSmooth
    global projectionMatrix
    SolidSmoothShader = extraFunctions.make_shader('shaders/SolidSmoothVertex.txt', 'shaders/SolidSmoothFragment.txt')

    glUseProgram(SolidSmoothShader)
    projectionMatrix_location_SolidSmooth = glGetUniformLocation(SolidSmoothShader, 'projectionMatrix')
    glUniformMatrix4fv(projectionMatrix_location_SolidSmooth, 1, GL_TRUE, projectionMatrix)

    viewMatrix_location_SolidSmooth = glGetUniformLocation(SolidSmoothShader, 'viewMatrix')
  def updateOpenGL(self):
    global SolidSmoothShader, pointVAO, vertexBuffer, pointVecs, indicesBuffer, triangles
    glUseProgram(SolidSmoothShader)
    glBindVertexArray(pointVAO)
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer)
    glBufferData(GL_ARRAY_BUFFER, pointVecs.nbytes, pointVecs, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indiceBuffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles.nbytes, triangles, GL_STATIC_DRAW)
  def calcPoints(self):
    global eigenstate_3d, triangles, pointVecs, occlusion
    
    filled = (eigenstate_3d**2) / (h**3) >= threshold

    nnn = filled[0:-1, 0:-1, 0:-1]
    pnn = filled[1:, 0:-1, 0:-1]
    npn = filled[0:-1, 1:, 0:-1]
    nnp = filled[0:-1, 0:-1, 1:]
    ppn = filled[1:, 1:, 0:-1]
    pnp = filled[1:, 0:-1, 1:]
    npp = filled[0:-1, 1:, 1:]
    ppp = filled[1:, 1:, 1:]

    corners = np.stack([nnn, pnn, npn, nnp, ppn, pnp, npp, ppp], axis=3)
    corner_counts = np.sum(corners, axis=3)
    considered = (corner_counts > 0) & (corner_counts < 8)
    
    zs, ys, xs = np.where(considered)

    corners_active = corners[zs, ys, xs]
    signs = np.sign(eigenstate_3d[zs, ys, xs])

    face_offsets = {}
    angles = [0,90,180,270,90,180,270,90,180,270,120,240,120,240,120,240,120,240,180,180,180,180,180,180]
    axes = [(1,0,0),(1,0,0),(1,0,0),(1,0,0),(0,1,0),(0,1,0),(0,1,0),(0,0,1),(0,0,1),(0,0,1),(1,1,1),(1,1,1),(1,1,-1),(1,1,-1),(-1,1,1),(-1,1,1),(-1,1,-1),(-1,1,-1),(1,-1,0),(1,1,0),(1,0,-1),(1,0,1),(0,1,1),(0,1,-1)]
    basicKey = [[-1,-1,-1],[-1,-1,1],[-1,1,-1],[1,-1,-1],[-1,1,1],[1,-1,1],[1,1,-1],[1,1,1]]

    def rotation(axis: tuple, point: list, angle: float) -> list:
      length = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
      normaxis = axis / length
      c, s, dot, cross = np.cos(angle), np.sin(angle), np.dot(normaxis, point), np.cross(normaxis, point)
      x = c * point[0] + (1 - c) * dot * normaxis[0] + s * cross[0]
      y = c * point[1] + (1 - c) * dot * normaxis[1] + s * cross[1]
      z = c * point[2] + (1 - c) * dot * normaxis[2] + s * cross[2]
      return [np.round(x, decimals=2), np.round(y, decimals=2), np.round(z, decimals=2)]

    def setting(index_key: list, number: int):
      target = np.array(index_key, dtype=bool)
      pos_match = np.all(corners_active == target, axis=1)
      neg_match = np.all(~corners_active == target, axis=1)
      
      mask = pos_match | neg_match
      
      chirality_val = np.where(pos_match[mask], 1, -1)
      chirality_block = np.repeat(chirality_val[:, np.newaxis], number, axis=1)
      return mask, chirality_block

    def addFace(name: str, key: list, number: int, indices: list, points: list):
      for index in indices:
        rad = angles[index] / 180 * np.pi
        tempkey = [rotation(axes[index], basicKey[id], rad) for id, val in enumerate(key) if val == 1]
        newkey = [coords in tempkey for coords in basicKey]
        newpoints = [[y + 0.5 for y in rotation(axes[index], [x - 0.5 for x in point], rad)] for point in points]
        
        mask, chir_block = setting(newkey, number)
        face_offsets[f'{name}{index}'] = (mask, chir_block, newpoints)

    addFace('Corner',[1,0,0,0,0,0,0,0],1,[0,1,2,4,5,7,8,18],[[0.0,0.5,0.0], [0.0,0.0,0.5], [0.5,0.0,0.0]])
    addFace('Edge',[1,1,0,0,0,0,0,0],2,[0,1,2,3,4,5,6,8,14,15,16,17],[[0.5,0.0,0.0],[0.0,0.5,0.0],[0.0,0.5,1.0],[0.5,0.0,0.0],[0.0,0.5,1.0],[0.5,0.0,1.0]])
    addFace('Face',[1,1,0,1,0,1,0,0],2,[0,1,7],[[0.0,0.5,0.0],[0.0,0.5,1.0],[1.0,0.5,0.0],[1.0,0.5,0.0],[0.0,0.5,1.0],[1.0,0.5,1.0]])
    addFace('Three',[1,1,1,0,0,0,0,0],3,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],[[0.5,0.0,1.0],[0.5,0.0,0.0],[0.5,1.0,0.0],[0.5,0.0,1.0],[0.5,1.0,0.0],[0.0,0.5,1.0],[0.5,1.0,0.0],[0.0,1.0,0.5],[0.0,0.5,1.0]])
    addFace('Hexagon',[1,1,1,1,0,0,0,0],4,[0,1,2,4,5,6,7,18],[[1.0,0.5,0.0],[0.5,1.0,0.0],[1.0,0.0,0.5],[1.0,0.0,0.5],[0.5,1.0,0.0],[0.0,1.0,0.5],[1.0,0.0,0.5],[0.0,1.0,0.5],[0.5,0.0,1.0],[0.5,0.0,1.0],[0.0,1.0,0.5],[0.0,0.5,1.0]])
    addFace('2Corner',[1,0,0,0,1,0,0,0],4,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],[[0.5,0.0,0.0],[0.0,0.5,0.0],[0.0,0.0,0.5],[0.0,1.0,0.5],[0.5,1.0,1.0],[0.0,0.5,1.0],[0.0,0.5,0.0],[0.0,0.5,0.5],[0.0,0.0,0.5],[0.0,0.5,0.5],[0.0,1.0,0.5],[0.0,0.5,1.0]])
    addFace('TwizzlerL',[1,1,1,0,0,0,1,0],4,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],[[0.5,0.0,1.0],[0.5,0.0,0.0],[0.0,0.5,1.0],[0.5,0.0,0.0],[1.0,1.0,0.5],[0.0,0.5,1.0],[0.0,0.5,1.0],[1.0,1.0,0.5],[0.0,1.0,0.5],[1.0,1.0,0.5],[0.5,0.0,0.0],[1.0,0.5,0.0]])
    addFace('TwizzlerD',[1,1,0,0,1,0,0,1],4,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],[[0.5,0.0,0.0],[0.0,0.5,0.0],[0.5,0.0,1.0],[1.0,1.0,0.5],[0.5,0.0,1.0],[0.0,0.5,0.0],[1.0,1.0,0.5],[0.0,0.5,0.0],[0.0,1.0,0.5],[1.0,1.0,0.5],[1.0,0.5,1.0],[0.5,0.0,1.0]])

    all_vertices = []
    all_chirality = []

    for faces, (exposed_mask, chir_block, offsets) in face_offsets.items():
      cx = xs[exposed_mask]
      cy = ys[exposed_mask]
      cz = zs[exposed_mask]
      csigns = signs[exposed_mask]
      
      centers = np.stack([cx, cy, cz, csigns], axis=-1)
      offsets_arr = np.array(offsets, dtype=np.float32)
      num_offsets = len(offsets_arr)
      
      face_verts = np.zeros((len(cx), num_offsets, 4), dtype=np.float32)
      face_verts[:, :, :3] = centers[:, np.newaxis, :3] + offsets_arr
      face_verts[:, :, 3] = centers[:, np.newaxis, 3]
      
      all_vertices.append(face_verts.reshape(-1, 4))
      all_chirality.append(chir_block.ravel())

    raw_points = np.vstack(all_vertices)
    chirality_arr = np.concatenate(all_chirality).astype(np.int32)
    
    scale_factors = np.array([h, h, h], dtype=np.float32)
    shift_factors = np.array([0.5 * h * N, 0.5 * h * N, 0.5 * h * N], dtype=np.float32)
    keys = raw_points[:, :3] * scale_factors - shift_factors
    keys = np.round(keys, decimals=6)
    
    _, unique_indices, inverse_indices = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    
    pointVecs = raw_points[unique_indices]
    pointVecs[:, :3] = (pointVecs[:, :3] * h) - (0.5 * h * N)

    new_indices = inverse_indices.reshape(-1, 3)
    
    flip_mask = (chirality_arr == -1)
    new_indices[flip_mask, 0], new_indices[flip_mask, 2] = (
        new_indices[flip_mask, 2].copy(),
        new_indices[flip_mask, 0].copy()
    )
    
    triangles = new_indices.ravel().astype(np.uint32)
  def draw(self):
    glUseProgram(SolidSmoothShader)
    glUniformMatrix4fv(viewMatrix_location_SolidSmooth, 1, GL_TRUE, viewMatrix)
    glBindVertexArray(pointVAO)
    glDrawElements(GL_TRIANGLES, len(triangles), GL_UNSIGNED_INT, ctypes.c_void_p(0))
  def changeEigen(self, change):
    global eigen, SolidSmoothShader, pointVAO, vertexBuffer, pointVecs, indiceBuffer, triangles, eigenvalues
    setUpFunctions.setUpPoints(h, N, eigen+change)
    eigen += change
    glUseProgram(SolidSmoothShader)
    glBindVertexArray(pointVAO)
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer)
    glBufferData(GL_ARRAY_BUFFER, pointVecs.nbytes, pointVecs, GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indiceBuffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles.nbytes, triangles, GL_STATIC_DRAW)
    display.set_caption(f'Waveform Render of Eigenvalue {eigen} at {eigenvalues[eigen-1]}')
  def changeRenderType(self):
    global renderType
    renderType = SolidGouraud()

    print(f'Start calcpoints at {time.perf_counter() - startTime:.2f} seconds')
    renderType.calcPoints()
    print(f'End calcpoints at {time.perf_counter() - startTime:.2f} seconds')
    renderType.updateOpenGL()

class SolidGouraud:
  @staticmethod
  def setUpOpenGL():
    global SolidGouraudShader, viewMatrix_location_SolidGouraud
    global projectionMatrix
    SolidGouraudShader = extraFunctions.make_shader('shaders/SolidGouraudVertex.txt', 'shaders/SolidGouraudFragment.txt')

    glUseProgram(SolidGouraudShader)
    projectionMatrix_location_SolidGouraud = glGetUniformLocation(SolidGouraudShader, 'projectionMatrix')
    glUniformMatrix4fv(projectionMatrix_location_SolidGouraud, 1, GL_TRUE, projectionMatrix)

    viewMatrix_location_SolidGouraud = glGetUniformLocation(SolidGouraudShader, 'viewMatrix')
  def updateOpenGL(self):
    global SolidGouraudShader, pointVAO, vertexBuffer, pointVecs, indicesBuffer, triangles
    glUseProgram(SolidGouraudShader)
    glBindVertexArray(pointVAO)
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer)
    glBufferData(GL_ARRAY_BUFFER, pointVecs.nbytes, pointVecs, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indiceBuffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles.nbytes, triangles, GL_STATIC_DRAW)
  def calcPoints(self):
    global eigenstate_3d, triangles, pointVecs, occlusion
    
    filled = (eigenstate_3d**2) / (h**3) >= threshold

    nnn = filled[0:-1, 0:-1, 0:-1]
    pnn = filled[1:, 0:-1, 0:-1]
    npn = filled[0:-1, 1:, 0:-1]
    nnp = filled[0:-1, 0:-1, 1:]
    ppn = filled[1:, 1:, 0:-1]
    pnp = filled[1:, 0:-1, 1:]
    npp = filled[0:-1, 1:, 1:]
    ppp = filled[1:, 1:, 1:]

    corners = np.stack([nnn, pnn, npn, nnp, ppn, pnp, npp, ppp], axis=3)
    corner_counts = np.sum(corners, axis=3)
    considered = (corner_counts > 0) & (corner_counts < 8)
    
    zs, ys, xs = np.where(considered)

    corners_active = corners[zs, ys, xs]
    signs = np.sign(eigenstate_3d[zs, ys, xs])

    face_offsets = {}
    angles = [0,90,180,270,90,180,270,90,180,270,120,240,120,240,120,240,120,240,180,180,180,180,180,180]
    axes = [(1,0,0),(1,0,0),(1,0,0),(1,0,0),(0,1,0),(0,1,0),(0,1,0),(0,0,1),(0,0,1),(0,0,1),(1,1,1),(1,1,1),(1,1,-1),(1,1,-1),(-1,1,1),(-1,1,1),(-1,1,-1),(-1,1,-1),(1,-1,0),(1,1,0),(1,0,-1),(1,0,1),(0,1,1),(0,1,-1)]
    basicKey = [[-1,-1,-1],[-1,-1,1],[-1,1,-1],[1,-1,-1],[-1,1,1],[1,-1,1],[1,1,-1],[1,1,1]]

    def rotation(axis: tuple, point: list, angle: float) -> list:
      length = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
      normaxis = axis / length
      c, s, dot, cross = np.cos(angle), np.sin(angle), np.dot(normaxis, point), np.cross(normaxis, point)
      x = c * point[0] + (1 - c) * dot * normaxis[0] + s * cross[0]
      y = c * point[1] + (1 - c) * dot * normaxis[1] + s * cross[1]
      z = c * point[2] + (1 - c) * dot * normaxis[2] + s * cross[2]
      return [np.round(x, decimals=2), np.round(y, decimals=2), np.round(z, decimals=2)]

    def setting(index_key: list, number: int):
      target = np.array(index_key, dtype=bool)
      pos_match = np.all(corners_active == target, axis=1)
      neg_match = np.all(~corners_active == target, axis=1)
      
      mask = pos_match | neg_match
      
      chirality_val = np.where(pos_match[mask], 1, -1)
      chirality_block = np.repeat(chirality_val[:, np.newaxis], number, axis=1)
      return mask, chirality_block

    def addFace(name: str, key: list, number: int, indices: list, points: list):
      for index in indices:
        rad = angles[index] / 180 * np.pi
        tempkey = [rotation(axes[index], basicKey[id], rad) for id, val in enumerate(key) if val == 1]
        newkey = [coords in tempkey for coords in basicKey]
        newpoints = [[y + 0.5 for y in rotation(axes[index], [x - 0.5 for x in point], rad)] for point in points]
        
        mask, chir_block = setting(newkey, number)
        face_offsets[f'{name}{index}'] = (mask, chir_block, newpoints)

    addFace('Corner',[1,0,0,0,0,0,0,0],1,[0,1,2,4,5,7,8,18],[[0.0,0.5,0.0], [0.0,0.0,0.5], [0.5,0.0,0.0]])
    addFace('Edge',[1,1,0,0,0,0,0,0],2,[0,1,2,3,4,5,6,8,14,15,16,17],[[0.5,0.0,0.0],[0.0,0.5,0.0],[0.0,0.5,1.0],[0.5,0.0,0.0],[0.0,0.5,1.0],[0.5,0.0,1.0]])
    addFace('Face',[1,1,0,1,0,1,0,0],2,[0,1,7],[[0.0,0.5,0.0],[0.0,0.5,1.0],[1.0,0.5,0.0],[1.0,0.5,0.0],[0.0,0.5,1.0],[1.0,0.5,1.0]])
    addFace('Three',[1,1,1,0,0,0,0,0],3,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],[[0.5,0.0,1.0],[0.5,0.0,0.0],[0.5,1.0,0.0],[0.5,0.0,1.0],[0.5,1.0,0.0],[0.0,0.5,1.0],[0.5,1.0,0.0],[0.0,1.0,0.5],[0.0,0.5,1.0]])
    addFace('Hexagon',[1,1,1,1,0,0,0,0],4,[0,1,2,4,5,6,7,18],[[1.0,0.5,0.0],[0.5,1.0,0.0],[1.0,0.0,0.5],[1.0,0.0,0.5],[0.5,1.0,0.0],[0.0,1.0,0.5],[1.0,0.0,0.5],[0.0,1.0,0.5],[0.5,0.0,1.0],[0.5,0.0,1.0],[0.0,1.0,0.5],[0.0,0.5,1.0]])
    addFace('2Corner',[1,0,0,0,1,0,0,0],4,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],[[0.5,0.0,0.0],[0.0,0.5,0.0],[0.0,0.0,0.5],[0.0,1.0,0.5],[0.5,1.0,1.0],[0.0,0.5,1.0],[0.0,0.5,0.0],[0.0,0.5,0.5],[0.0,0.0,0.5],[0.0,0.5,0.5],[0.0,1.0,0.5],[0.0,0.5,1.0]])
    addFace('TwizzlerL',[1,1,1,0,0,0,1,0],4,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],[[0.5,0.0,1.0],[0.5,0.0,0.0],[0.0,0.5,1.0],[0.5,0.0,0.0],[1.0,1.0,0.5],[0.0,0.5,1.0],[0.0,0.5,1.0],[1.0,1.0,0.5],[0.0,1.0,0.5],[1.0,1.0,0.5],[0.5,0.0,0.0],[1.0,0.5,0.0]])
    addFace('TwizzlerD',[1,1,0,0,1,0,0,1],4,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],[[0.5,0.0,0.0],[0.0,0.5,0.0],[0.5,0.0,1.0],[1.0,1.0,0.5],[0.5,0.0,1.0],[0.0,0.5,0.0],[1.0,1.0,0.5],[0.0,0.5,0.0],[0.0,1.0,0.5],[1.0,1.0,0.5],[1.0,0.5,1.0],[0.5,0.0,1.0]])

    all_vertices = []
    all_chirality = []

    for faces, (exposed_mask, chir_block, offsets) in face_offsets.items():
      cx = xs[exposed_mask]
      cy = ys[exposed_mask]
      cz = zs[exposed_mask]
      csigns = signs[exposed_mask]
      
      centers = np.stack([cx, cy, cz, csigns], axis=-1)
      offsets_arr = np.array(offsets, dtype=np.float32)
      num_offsets = len(offsets_arr)
      
      face_verts = np.zeros((len(cx), num_offsets, 4), dtype=np.float32)
      face_verts[:, :, :3] = centers[:, np.newaxis, :3] + offsets_arr
      face_verts[:, :, 3] = centers[:, np.newaxis, 3]
      
      all_vertices.append(face_verts.reshape(-1, 4))
      all_chirality.append(chir_block.ravel())

    raw_points = np.vstack(all_vertices)
    chirality_arr = np.concatenate(all_chirality).astype(np.int32)
    
    scale_factors = np.array([h, h, h], dtype=np.float32)
    shift_factors = np.array([0.5 * h * N, 0.5 * h * N, 0.5 * h * N], dtype=np.float32)
    keys = raw_points[:, :3] * scale_factors - shift_factors
    keys = np.round(keys, decimals=6)
    
    _, unique_indices, inverse_indices = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    
    pointVecs = raw_points[unique_indices]
    pointVecs[:, :3] = (pointVecs[:, :3] * h) - (0.5 * h * N)

    new_indices = inverse_indices.reshape(-1, 3)
    
    flip_mask = (chirality_arr == -1)
    new_indices[flip_mask, 0], new_indices[flip_mask, 2] = (
        new_indices[flip_mask, 2].copy(),
        new_indices[flip_mask, 0].copy()
    )
    
    triangles = new_indices.ravel().astype(np.uint32)
  def draw(self):
    glUseProgram(SolidGouraudShader)
    glUniformMatrix4fv(viewMatrix_location_SolidGouraud, 1, GL_TRUE, viewMatrix)
    glBindVertexArray(pointVAO)
    glDrawElements(GL_TRIANGLES, len(triangles), GL_UNSIGNED_INT, ctypes.c_void_p(0))
  def changeEigen(self, change):
    global eigen, SolidGouraudShader, pointVAO, vertexBuffer, pointVecs, indiceBuffer, triangles, eigenvalues
    setUpFunctions.setUpPoints(h, N, eigen+change)
    eigen += change
    glUseProgram(SolidGouraudShader)
    glBindVertexArray(pointVAO)
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer)
    glBufferData(GL_ARRAY_BUFFER, pointVecs.nbytes, pointVecs, GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indiceBuffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles.nbytes, triangles, GL_STATIC_DRAW)
    display.set_caption(f'Waveform Render of Eigenvalue {eigen} at {eigenvalues[eigen-1]}')
  def changeRenderType(self):
    global renderType
    renderType = Fishnet()

    print(f'Start calcpoints at {time.perf_counter() - startTime:.2f} seconds')
    renderType.calcPoints()
    print(f'End calcpoints at {time.perf_counter() - startTime:.2f} seconds')
    renderType.updateOpenGL()


renderType = SolidGouraud()


class extraFunctions:
  @staticmethod
  def make_shader(vertex_filename: str, fragment_filename: str) -> int:
    vertex_module = extraFunctions.make_shader_module(vertex_filename, GL_VERTEX_SHADER)
    fragment_module = extraFunctions.make_shader_module(fragment_filename, GL_FRAGMENT_SHADER)
    return compileProgram(vertex_module, fragment_module)
  @staticmethod
  def make_shader_module(filename: str, module_type: int) -> int:
    with open(filename, "r") as file:
      source_code = file.readlines()
      return compileShader(source_code, module_type)
  @staticmethod
  def get_shifted_mask(mask, dx, dy, dz):
    return np.roll(np.roll(np.roll(mask, dx, axis=2), dy, axis=1), dz, axis=0)

class setUpFunctions:
  @staticmethod
  def setUpUI():
    global interfaceShader, indices, interfaceVAO

    interfaceShader = extraFunctions.make_shader('shaders/interfaceVertex.txt', 'shaders/interfaceFragment.txt')
    glUseProgram(interfaceShader)

    interfaceVAO = glGenVertexArrays(1)
    glBindVertexArray(interfaceVAO)
  @staticmethod
  def setUpPoints(h: float, N: int, eigen:int):
    global threshold, eigenstate, eigenvalues, eigenstate_3d

    eigenvalues = np.load('eigenvalues.npy')
    eigenstate = np.load('eigenstates.npy')[:, eigen-1]

    sortedEigenstate = np.sort(eigenstate**2, descending=True)
    low = 0
    high = 1
    unfound = True
    while unfound:
      nextSum = sum(sortedEigenstate[:np.rint((low+high)/2*len(sortedEigenstate)).astype(int)])
      if nextSum > totalProb*0.95 and nextSum < totalProb*1.05:
        high = (low+high)/2
        low = (low+high)/2
        unfound = False
      elif nextSum < totalProb:
        low = (low+high)/2
      else:
        high = (low+high)/2
    threshold = sortedEigenstate[np.rint((low+high)/2*len(sortedEigenstate)).astype(int)]/h**3

    eigenstate_3d = eigenstate.reshape((N, N, N))

    print(f'Start calcpoints at {time.perf_counter() - startTime:.2f} seconds')
    renderType.calcPoints()
    print(f'End calcpoints at {time.perf_counter() - startTime:.2f} seconds')

  @staticmethod
  def setUpRender(eigen:int, ar: float, fov: float, zn: float, zf: float):
    global rVec, uVec, fVec, camVec, a1, a2, a3
    global viewMatrix, clock, pointVAO, screen
    global viewMatrix_location_Fishnet, viewMatrix_location_SolidFlat, viewMatrix_location_SolidShaded
    global FishnetShader, SolidFlatShader, SolidShadedShader
    global pointVecs, triangles, occlusion
    global vertexBuffer, indiceBuffer, occlusionBuffer
    global projectionMatrix

    display.init()

    display.gl_set_attribute(locals.GL_CONTEXT_MAJOR_VERSION, 3)
    display.gl_set_attribute(locals.GL_CONTEXT_MINOR_VERSION, 1)
    display.gl_set_attribute(locals.GL_CONTEXT_PROFILE_MASK, locals.GL_CONTEXT_PROFILE_CORE)
    display.set_caption(f'Waveform Render of Eigenvalue {eigen} at {eigenvalues[eigen-1]}')

    screen = display.set_mode(SCREEN_SIZE, WINDOW_CREATION_FLAGS)
    clock = pygame.time.Clock()

    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_PROGRAM_POINT_SIZE)

    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glFrontFace(GL_CCW)

    projectionMatrix = np.array([
      [fov/ar, 0, 0, 0],
      [0, fov, 0, 0],
      [0, 0, (zf+zn)/(zn-zf), 2*zf*zn/(zn-zf)],
      [0, 0, -1, 0]
    ], dtype=np.float32)

    rVec = np.array([0, 1, 0], dtype=np.float32)
    uVec = np.array([0, 0, 1], dtype=np.float32)
    fVec = np.array([-1, 0, 0], dtype=np.float32)
    camVec = np.array([-5.0, 0.0, 0.0], dtype=np.float32)

    a1 = 0
    a2 = 0
    a3 = 0

    viewMatrix = np.array([
      [rVec[0], rVec[1], rVec[2], -np.dot(rVec, camVec)],
      [uVec[0], uVec[1], uVec[2], -np.dot(uVec, camVec)],
      [fVec[0], fVec[1], fVec[2], -np.dot(fVec, camVec)],
      [0, 0, 0, 1]
    ], dtype=np.float32)

    Fishnet.setUpOpenGL()
    SolidFlat.setUpOpenGL()
    SolidShaded.setUpOpenGL()
    SolidSmooth.setUpOpenGL()
    SolidGouraud.setUpOpenGL()

    pointVAO = glGenVertexArrays(1)
    glBindVertexArray(pointVAO)

    vertexBuffer, indiceBuffer, occlusionBuffer = glGenBuffers(3)

    renderType.updateOpenGL()

class mathFunctions:
  @staticmethod
  def wavefunction(x: float, y: float, z: float, a: float, i: int, j: int, k: int) -> float:
    return (x**i)*(y**j)*(z**k)*np.exp(-a*(x**2+y**2+z**2))
  @staticmethod
  def extPotential(x: float, y: float, z:float) -> float:
    return -1/np.sqrt(x**2 + y**2 + z**2)
  @staticmethod
  def calcEigens(h: float, h2: float, N: int, eigens: int):
    if os.path.exists('eigenvalues.npy'):
      os.remove('eigenvalues.npy')
    if os.path.exists('eigenstates.npy'):
      os.remove('eigenstates.npy')

    matrixT = sparse.diags_array([6/h2, -1/h2, -1/h2, -1/h2, -1/h2, -1/h2, -1/h2], offsets=[0, N**2, -N**2, 1, -1, N, -N], dtype=None, shape=(N**3, N**3))

    potentials: list[float] = []
    potentials.extend([0 for i in range(N*N)])
    for Z in range(N-2):
      for Y in range(N):
        for X in range(N):
          potentials.append(mathFunctions.extPotential(h*X-h/2*(N-1), h*Y-h/2*(N-1), h*(Z+1)-h/2*(N-1)))
    potentials.extend([0 for i in range(N*N)])
    matrixV = sparse.diags_array([potentials], offsets=[0], dtype=None, shape=(N**3, N**3))

    boundaries: list[int] = []
    boundaries.extend([0 for i in range(N*N)])
    for a in range(N-2):
      boundaries.extend([0 for i in range(N)])
      for b in range(N-2):
        boundaries.append(0)
        boundaries.extend([1 for i in range(N-2)])
        boundaries.append(0)
      boundaries.extend([0 for i in range(N)])
    boundaries.extend([0 for i in range(N*N)])
    matrixB = sparse.diags_array([boundaries], offsets=[0], dtype=None, shape=(N**3, N**3))

    matrixAll = (matrixB @ (matrixT + matrixV)) @ matrixB

    eigenvalues, eigenstates = sparse.linalg.eigsh(matrixAll, k=eigens, which='SM')

    np.save('eigenvalues.npy', eigenvalues)
    np.save('eigenstates.npy', eigenstates)

if calculate:
  mathFunctions.calcEigens(h, h**2, N, eigens)

if render:
  setUpFunctions.setUpPoints(h, N, eigen)
  setUpFunctions.setUpRender(eigen, SCREEN_SIZE[0]/SCREEN_SIZE[1], 1/np.tan(np.pi/2/2), 0.1, 100)
  setUpFunctions.setUpUI()

  print(f'Finish setup at {time.perf_counter() - startTime:.2f} seconds')

  running = True
  while running:
    for e in event.get():
      if e.type == locals.QUIT:
        print(f'Quit at {time.perf_counter() - startTime:.2f} seconds')
        running = False
      elif e.type == pygame.MOUSEBUTTONDOWN:
        if SCREEN_SIZE[0]/20 < e.pos[0] < SCREEN_SIZE[0]/10 and SCREEN_SIZE[1]/20 < e.pos[1] < SCREEN_SIZE[1]/10 and eigen > 1:
          renderType.changeEigen(-1)
        elif 9*SCREEN_SIZE[0]/10 < e.pos[0] < 19*SCREEN_SIZE[0]/20 and SCREEN_SIZE[1]/20 < e.pos[1] < SCREEN_SIZE[1]/10 and eigen < eigens:
          renderType.changeEigen(1)
        elif 2*SCREEN_SIZE[0]/5 < e.pos[0] < 3*SCREEN_SIZE[0]/5 and SCREEN_SIZE[1]/20 < e.pos[1] < SCREEN_SIZE[1]/10:
          renderType.changeRenderType()

    keys = key.get_pressed()

    if keys[locals.K_a]:
      camVec += 0.05*np.array([np.cos(a1 + np.pi/2), np.sin(a1 + np.pi/2), 0], dtype=np.float32)
    if keys[locals.K_d]:
      camVec += -0.05*np.array([np.cos(a1 + np.pi/2), np.sin(a1 + np.pi/2), 0], dtype=np.float32)
    if keys[locals.K_w]:
      camVec += 0.05* np.array([np.cos(a1),np.sin(a1), 0], dtype=np.float32)
    if keys[locals.K_s]:
      camVec += -0.05* np.array([np.cos(a1),np.sin(a1), 0], dtype=np.float32)
    if keys[locals.K_SPACE]:
      camVec[2] += 0.05
    if keys[locals.K_LSHIFT]:
      camVec[2] += -0.05
    if keys[locals.K_LEFT]:
      a1 += 0.01
    if keys[locals.K_RIGHT]:
      a1 += -0.01
    if keys[locals.K_UP]:
      a2 = np.max([-np.pi/2, a2 - 0.01])
    if keys[locals.K_DOWN]:
      a2 = np.min([np.pi/2, a2 + 0.01])

    rVec = np.array([-np.cos(a1 + np.pi/2), -np.sin(a1 + np.pi/2), 0], dtype=np.float32)
    uVec = np.array([-np.cos(a1) * np.cos(a2 + np.pi/2),-np.sin(a1) * np.cos(a2 + np.pi/2), np.sin(a2 + np.pi/2)], dtype=np.float32)
    fVec = np.array([-np.cos(a1) * np.cos(a2),-np.sin(a1) * np.cos(a2), np.sin(a2)], dtype=np.float32)

    viewMatrix = np.array([
      [rVec[0], rVec[1], rVec[2], -np.dot(rVec, camVec)],
      [uVec[0], uVec[1], uVec[2], -np.dot(uVec, camVec)],
      [fVec[0], fVec[1], fVec[2], -np.dot(fVec, camVec)],
      [0, 0, 0, 1]
    ], dtype=np.float32)

    glClear(GL_COLOR_BUFFER_BIT)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    renderType.draw()

    glUseProgram(interfaceShader)
    glBindVertexArray(interfaceVAO)
    glDrawArrays(GL_TRIANGLES, 0, 18)

    display.flip()
    clock.tick(FRAMERATE)

  glDeleteVertexArrays(2, [pointVAO, interfaceVAO])
  glDeleteBuffers(3, [vertexBuffer, indiceBuffer, occlusionBuffer])
  glDeleteProgram(FishnetShader)
  glDeleteProgram(SolidFlatShader)
  glDeleteProgram(SolidShadedShader)
  glDeleteProgram(SolidSmoothShader)
  glDeleteProgram(SolidGouraudShader)
  glDeleteProgram(interfaceShader)
  quit()
