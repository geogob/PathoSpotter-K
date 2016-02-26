# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 20:55:02 2016

@author: George Oliveira Barros, george_gob@hotmail.com. 
PGCA - Universidade Estadual de Feira de Santana (UEFS)

PathoSpoter-K Classification: Version 1.0
"""

print __doc__
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation, metrics
#from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
import numpy as np

listaLabels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
listaQntRegions = [183, 238, 240, 225, 228, 176, 232, 211, 247, 226, 213, 214, 248, 387, 214, 177, 330, 249, 210, 267, 251, 263, 280, 221, 174, 142, 140, 160, 174, 215, 198, 169, 216, 228, 218, 245, 249, 215, 238, 224, 252, 220, 209, 194, 104, 104, 204, 304, 195, 153, 232, 190, 186, 174, 219, 201, 251, 202, 171, 204, 194, 189, 189, 187, 193, 265, 171, 102, 219, 168, 176, 197, 201, 88, 211, 220, 166, 199, 213, 172, 177, 221, 196, 237, 232, 160, 122, 188, 190, 195, 200, 254, 237, 194, 195, 206, 173, 92, 148, 193, 192, 155, 212, 244, 208, 224, 257, 342, 260, 332, 316, 307, 92, 124, 156, 103, 95, 217, 193, 247, 199, 237, 252, 238, 205, 214, 300, 241, 206, 198, 206, 195, 217, 196, 176, 194, 294, 259, 312, 261, 322, 293, 316, 263, 203, 203, 173, 242, 262, 243, 288, 263, 264, 159, 253, 259, 209, 227, 251, 230, 242, 334, 277, 220, 195, 233, 172, 279, 198, 164, 182, 161, 181, 188, 227, 219, 223, 227, 198, 225, 172, 238, 187, 194, 203, 433, 216, 194, 240, 248, 229, 204, 153, 187, 156, 207, 155, 202, 232, 231, 475, 214, 193, 179, 191, 196, 206, 232, 180, 303, 163, 204, 224, 232, 195, 304, 113, 251, 312, 307, 196, 307, 326, 292, 270, 246, 310, 243, 292, 270, 219, 342, 241, 128, 368, 198, 263, 90, 242, 212, 162, 181, 255, 292, 266, 257, 213, 316, 221, 354, 314, 289, 259, 252, 308, 316, 275, 302, 228, 183, 293, 219, 266, 258, 264, 209, 308, 252, 349, 328, 235, 221, 154, 243, 310, 239, 243, 229, 208, 199, 195, 228, 180, 272, 266, 247, 153, 209, 184, 207, 341, 350, 388, 174, 344, 397, 390, 314, 223, 99, 121, 94, 93, 92, 101, 183, 174, 132, 153, 144, 121, 101, 141, 202, 152, 190, 182, 147, 99, 118, 255, 166, 139, 82, 76, 253, 217, 201, 130, 205, 287, 298, 265, 179, 235, 194, 287, 270, 313, 269, 249, 101, 240, 249, 182, 198, 140, 114, 315, 283, 300, 313, 303, 195, 110, 113, 124, 139, 192, 223, 199, 81, 173, 129, 125, 157, 146, 239, 167, 197, 120, 128, 170, 165, 107, 136, 105, 205, 239, 132, 136, 127, 119, 158, 267, 149, 268, 197, 160, 178, 200, 174, 244, 236, 202, 213, 171, 216, 219, 203, 236, 156, 133, 267, 251, 231, 263, 280, 142, 181, 228, 218, 245, 249, 215, 238, 224, 247, 203, 165, 204, 207, 213, 232, 190, 189, 239, 191, 213, 220, 197, 201, 237, 232, 188, 243, 254, 237, 194, 195, 206, 204, 192, 155, 212, 185, 178, 192, 238, 170, 236, 224, 116, 118, 144, 202, 198, 195, 183, 133, 99, 158, 187, 329, 237, 334, 146, 130, 216, 201, 218, 241, 242, 193, 298, 99, 93, 138, 98, 113, 190, 128, 182, 118, 133, 269, 242, 372, 342, 302, 287, 313, 269, 172, 232, 324, 201, 140, 114, 167, 139, 128, 176, 186, 127, 110, 151, 203, 102, 160, 72, 112, 154, 132, 114, 108, 103, 133, 130, 103, 126, 91, 152, 133, 170, 184, 209, 136, 124, 95, 101, 151, 191, 118, 114, 90, 113, 149, 157, 114, 102, 117, 126, 168, 154, 125, 103, 164, 99, 161, 144, 70, 100, 166, 162, 138, 173, 133, 206, 188, 152, 105, 118, 101, 133, 202, 154, 114, 111, 121, 150, 132, 114, 147, 121, 149, 110, 90, 94, 130, 140, 145, 50, 158, 180, 168, 127, 145, 127, 120, 106, 105, 144, 130, 161, 130, 137, 123, 101, 107, 123, 139, 116, 127, 134, 105, 138, 107, 113, 117, 140, 121, 149, 114, 151, 155, 149, 150, 149, 137, 134, 135, 137, 145, 144, 133, 117, 131, 156, 188, 116, 141, 136, 116, 118, 121, 106, 120, 133, 135, 118, 115, 130, 127, 171, 178, 139, 121, 134, 103, 134, 102, 166, 137, 99, 91, 108, 91, 77, 180, 140, 147, 92, 107, 98, 116, 108, 43, 135, 143, 115, 103, 147, 119, 108, 108, 93, 108, 161, 117, 104, 110, 74, 105, 102, 70, 103, 96, 92, 85, 106, 52, 95, 121, 89, 120, 90, 33, 79, 89, 138, 147, 102, 87, 135, 128, 119, 89, 139, 88, 116, 113, 133, 147, 151, 140, 97, 115, 66, 103, 127, 124, 125, 113, 144, 147, 167, 107, 172, 100, 86, 116, 103, 126, 109, 123, 102, 103, 72, 72, 116, 110, 91, 130, 91, 73, 99, 127, 131, 109, 88, 84, 132, 155, 130, 131, 123, 138, 95, 118, 128, 113, 127, 120, 82, 91, 101, 131, 110, 97, 106, 102, 113, 117, 106, 129, 103, 116, 117, 98, 122, 112, 162, 70, 63, 133, 101, 110, 130, 125, 43, 74, 132, 133, 105, 34, 102, 77, 103, 81, 42, 138, 78, 144, 80, 82, 77, 157, 90, 120, 123]
listaQntBlobs = [234, 220, 214, 226, 240, 182, 215, 222, 212, 235, 231, 185, 225, 173, 177, 167, 263, 206, 194, 281, 268, 262, 281, 247, 205, 147, 138, 152, 164, 197, 210, 171, 180, 241, 229, 264, 277, 238, 239, 226, 227, 233, 210, 228, 143, 106, 230, 300, 200, 162, 274, 213, 227, 164, 239, 230, 244, 166, 151, 206, 237, 215, 217, 209, 244, 258, 184, 129, 242, 211, 202, 215, 211, 91, 218, 254, 208, 192, 232, 181, 217, 260, 221, 253, 251, 143, 153, 228, 239, 244, 242, 277, 249, 236, 223, 232, 199, 105, 204, 227, 211, 205, 255, 224, 233, 260, 261, 292, 261, 262, 301, 300, 97, 116, 147, 113, 90, 181, 242, 250, 197, 245, 236, 260, 217, 217, 243, 247, 215, 224, 214, 208, 243, 239, 196, 224, 264, 236, 253, 260, 264, 284, 288, 248, 226, 201, 204, 256, 245, 245, 284, 254, 283, 158, 239, 241, 230, 250, 233, 238, 242, 277, 241, 251, 241, 238, 206, 258, 224, 190, 208, 198, 213, 225, 269, 243, 220, 218, 204, 256, 198, 211, 192, 207, 224, 350, 230, 205, 262, 266, 263, 213, 166, 219, 158, 212, 188, 213, 208, 247, 294, 236, 227, 207, 220, 209, 210, 240, 223, 239, 196, 223, 247, 234, 216, 247, 124, 228, 271, 266, 225, 246, 259, 281, 222, 254, 242, 244, 278, 252, 230, 253, 252, 114, 288, 238, 236, 112, 269, 219, 206, 219, 229, 240, 244, 252, 223, 290, 230, 265, 261, 258, 238, 216, 256, 257, 227, 252, 243, 200, 250, 217, 272, 245, 275, 255, 260, 241, 252, 252, 252, 212, 172, 217, 277, 222, 260, 241, 217, 240, 215, 240, 216, 270, 264, 248, 181, 230, 215, 215, 298, 305, 316, 203, 328, 327, 345, 307, 241, 91, 113, 87, 102, 83, 101, 106, 114, 87, 72, 66, 103, 111, 55, 72, 59, 122, 133, 126, 102, 98, 123, 97, 83, 75, 81, 131, 223, 226, 112, 201, 247, 283, 247, 150, 196, 237, 277, 266, 316, 265, 219, 103, 251, 121, 90, 145, 107, 101, 317, 309, 333, 348, 339, 108, 120, 91, 108, 138, 170, 177, 176, 97, 94, 85, 89, 94, 101, 151, 136, 157, 108, 158, 131, 151, 85, 130, 100, 177, 212, 77, 75, 114, 61, 143, 210, 135, 152, 149, 124, 121, 127, 70, 232, 199, 209, 231, 144, 178, 189, 201, 185, 153, 104, 281, 268, 236, 262, 281, 147, 211, 241, 229, 264, 277, 238, 239, 226, 240, 222, 199, 230, 231, 221, 274, 213, 215, 206, 184, 240, 215, 215, 211, 253, 251, 228, 231, 277, 249, 236, 223, 232, 216, 211, 205, 255, 204, 218, 223, 254, 232, 262, 227, 178, 157, 197, 242, 244, 232, 216, 186, 141, 201, 226, 349, 332, 359, 207, 187, 235, 253, 263, 278, 264, 256, 256, 82, 102, 71, 111, 102, 122, 63, 133, 98, 122, 156, 247, 371, 338, 340, 277, 316, 265, 132, 196, 305, 184, 107, 101, 100, 138, 158, 145, 174, 114, 52, 142, 163, 99, 124, 87, 144, 174, 139, 86, 79, 117, 144, 119, 153, 100, 85, 137, 122, 168, 175, 190, 190, 139, 89, 104, 161, 185, 104, 110, 101, 136, 119, 148, 121, 119, 97, 100, 160, 151, 126, 109, 168, 125, 158, 138, 89, 94, 156, 114, 138, 161, 169, 202, 206, 202, 133, 153, 143, 167, 272, 182, 147, 154, 171, 191, 176, 156, 186, 183, 187, 163, 138, 142, 172, 183, 186, 74, 191, 211, 206, 164, 163, 163, 146, 140, 139, 191, 171, 197, 136, 168, 155, 148, 149, 159, 185, 147, 183, 159, 142, 171, 155, 142, 147, 185, 157, 195, 160, 199, 200, 181, 210, 199, 178, 171, 168, 176, 175, 154, 170, 163, 164, 188, 197, 157, 184, 165, 159, 151, 145, 141, 160, 168, 183, 150, 166, 148, 121, 168, 166, 190, 128, 125, 124, 127, 139, 156, 145, 93, 94, 113, 87, 73, 173, 159, 156, 113, 103, 84, 80, 62, 17, 141, 151, 78, 43, 143, 138, 61, 71, 72, 61, 168, 117, 103, 129, 65, 87, 102, 76, 132, 92, 90, 74, 90, 52, 128, 118, 91, 109, 101, 35, 72, 72, 132, 142, 76, 70, 140, 141, 138, 116, 141, 102, 133, 134, 153, 155, 149, 155, 101, 132, 67, 118, 143, 129, 135, 139, 134, 150, 153, 95, 149, 131, 108, 139, 121, 123, 132, 141, 126, 112, 85, 94, 188, 128, 110, 121, 123, 93, 96, 130, 138, 104, 98, 91, 141, 167, 144, 147, 115, 159, 96, 127, 141, 115, 121, 119, 96, 104, 120, 143, 125, 106, 109, 101, 118, 122, 108, 131, 99, 127, 123, 89, 125, 132, 159, 77, 68, 139, 111, 120, 140, 135, 44, 74, 110, 140, 112, 35, 95, 85, 128, 102, 26, 88, 69, 145, 73, 86, 77, 162, 81, 64, 73]
listaDensidade = [0.14259974161783853, 0.10240012519918051, 0.11631393432617188, 0.12925974527994791, 0.1241162618001302, 0.10955301920572917, 0.11319351196289063, 0.12614186604817709, 0.10353292099837698, 0.12594477335611978, 0.12879308064778647, 0.10716622223694452, 0.11925955550477062, 0.084521174561015197, 0.09967966062386456, 0.12277371936477662, 0.13152313232421875, 0.10944059995154885, 0.12099886407935403, 0.13433456420898438, 0.14022318522135416, 0.13491058349609375, 0.13817596435546875, 0.13074366251627603, 0.14420445760091147, 0.1470184326171875, 0.12198301010042495, 0.12066718902615733, 0.12201584012859502, 0.11794183297778583, 0.11299052299052299, 0.10998516387177988, 0.11834477051868356, 0.13587443033854166, 0.13272984822591147, 0.13988876342773438, 0.13396708170572916, 0.1309814453125, 0.1322174072265625, 0.13536453247070313, 0.12888115631691649, 0.13497034708658853, 0.1273092962320552, 0.11650975545247395, 0.13344006418990523, 0.13238078512510085, 0.12782541910807291, 0.135498046875, 0.13477356989537834, 0.14321194079312805, 0.14190165201822916, 0.13313166300455728, 0.14288075764973959, 0.10593983146246644, 0.13647969563802084, 0.13041178385416666, 0.13909022013346353, 0.099336074983536166, 0.099481142147679669, 0.10854625179188802, 0.13561630249023438, 0.13582738240559897, 0.13300196329752603, 0.12707392374674478, 0.13711802164713541, 0.14165496826171875, 0.13724009195963541, 0.1420745849609375, 0.13221359252929688, 0.13547388712565103, 0.12723668416341147, 0.12846501668294272, 0.12778981526692709, 0.11436417670013471, 0.13105138142903647, 0.13408025105794272, 0.1240997314453125, 0.1233990987141927, 0.12305704752604167, 0.13105901082356772, 0.135589599609375, 0.13886769612630209, 0.1295318603515625, 0.13025792439778647, 0.14262517293294272, 0.11429672827913413, 0.14017784598718064, 0.14192708333333334, 0.13170878092447916, 0.12916692097981772, 0.13143030802408853, 0.13939921061197916, 0.13486353556315103, 0.13083775838216147, 0.1463623046875, 0.13649749755859375, 0.12887700398763022, 0.13393529256184897, 0.14180501302083334, 0.13288625081380209, 0.13013839721679688, 0.13736979166666666, 0.13646570841471353, 0.1119391971664699, 0.13927332560221353, 0.13922627766927084, 0.13683446248372397, 0.13519922892252603, 0.13685480753580728, 0.13636906941731772, 0.13392766316731772, 0.13658523559570313, 0.11534771388870509, 0.10358450154368522, 0.10934278718173271, 0.14439506561266896, 0.102228532922603, 0.099419448476052247, 0.14734395345052084, 0.14086659749348959, 0.13931528727213541, 0.13874308268229166, 0.13080134691586878, 0.12962722778320313, 0.115081787109375, 0.1160875956217448, 0.13528188069661459, 0.13269297281901041, 0.11394138677130915, 0.12903467814127603, 0.13259506225585938, 0.12315750122070313, 0.12940216064453125, 0.12910207112630209, 0.11796061197916667, 0.12636184692382813, 0.13224919637044272, 0.11652119954427083, 0.12354405721028645, 0.13604736328125, 0.13016128540039063, 0.128875732421875, 0.12930425008138022, 0.131500244140625, 0.11751556396484375, 0.13300196329752603, 0.12395858764648438, 0.13697179158528647, 0.13598378499348959, 0.13272221883138022, 0.14254379272460938, 0.13717015584309897, 0.14274851481119791, 0.11142985026041667, 0.1154925028483073, 0.13084920247395834, 0.13701756795247397, 0.12750371297200522, 0.12594477335611978, 0.12856419881184897, 0.13707987467447916, 0.10233688354492187, 0.11424949759373842, 0.14464441935221353, 0.14053726196289063, 0.14427566528320313, 0.12519709269205728, 0.13973236083984375, 0.13189442952473959, 0.12770970662434897, 0.12938944498697916, 0.11589431762695313, 0.12417348225911458, 0.12478892008463542, 0.12941996256510416, 0.12869771321614584, 0.12907282511393228, 0.13402430216471353, 0.13164393107096353, 0.13706715901692709, 0.12534713745117188, 0.10640716552734375, 0.1187451680501302, 0.12378438313802083, 0.12224960327148438, 0.12111663818359375, 0.13001124064127603, 0.13206609090169272, 0.14950180053710938, 0.14963531494140625, 0.14352798461914063, 0.13455835978190103, 0.13177998860677084, 0.13243738810221353, 0.12258021036783855, 0.13807042439778647, 0.13365936279296875, 0.11696751912434895, 0.1220245361328125, 0.13520050048828125, 0.13077545166015625, 0.12591044108072916, 0.12874730428059897, 0.12243398030598958, 0.11942036946614583, 0.11842600504557292, 0.12164688110351563, 0.12264506022135417, 0.12557220458984375, 0.10599263509114583, 0.11997222900390625, 0.14286677042643228, 0.14405695597330728, 0.1233062744140625, 0.12920506795247397, 0.12449137369791667, 0.14080429077148438, 0.12990570068359375, 0.11734390258789063, 0.12002309163411458, 0.13072077433268228, 0.12396240234375, 0.13230641682942709, 0.13964462280273438, 0.12377675374348958, 0.12936019897460938, 0.12225723266601563, 0.1217053731282552, 0.13095219930013022, 0.13232549031575522, 0.13696161905924478, 0.12924321492513022, 0.12205759684244792, 0.13169987996419272, 0.13056564331054688, 0.12679926554361978, 0.12138621012369792, 0.12267176310221355, 0.13043467203776041, 0.1185925801595052, 0.11587905883789063, 0.11570231119791667, 0.1067339579264323, 0.12091827392578125, 0.12338002522786458, 0.1108385721842448, 0.11268107096354167, 0.12883122762044272, 0.12491226196289063, 0.11407979329427083, 0.10219701131184895, 0.11516571044921875, 0.13036855061848959, 0.12711842854817709, 0.11739095052083333, 0.12369410196940105, 0.1088244120279948, 0.08798980712890625, 0.13118871053059897, 0.11565144856770833, 0.1149444580078125, 0.1182543436686198, 0.12915420532226563, 0.11673227945963542, 0.12949117024739584, 0.14134470621744791, 0.13586171468098959, 0.12184906005859375, 0.13877360026041666, 0.11688486735026042, 0.13666025797526041, 0.12707392374674478, 0.1222394307454427, 0.12258275349934895, 0.12906010945638022, 0.11810684204101563, 0.13094711303710938, 0.11615626017252605, 0.12779617309570313, 0.14142227172851563, 0.13118998209635416, 0.11801656087239583, 0.11808268229166667, 0.13940938313802084, 0.13597615559895834, 0.12727610270182291, 0.12292989095052083, 0.123199462890625, 0.12709172566731772, 0.12866465250651041, 0.13984171549479166, 0.14039738972981772, 0.13593546549479166, 0.14062372843424478, 0.13998794555664063, 0.14150492350260416, 0.14250055948893228, 0.13884226481119791, 0.14399464925130209, 0.13731294250160911, 0.13598543767431881, 0.13471492169062432, 0.13087320317528428, 0.13464787599227634, 0.1327672441536151, 0.094970882023988581, 0.08122100530871873, 0.081176718274276097, 0.12200586362095214, 0.089106040451584295, 0.14040374919545162, 0.13430259064578418, 0.053644086742678294, 0.050330232370826788, 0.08838838395034361, 0.13812825520833333, 0.13833658854166667, 0.14691723878995924, 0.14040039691053421, 0.13423889723235358, 0.090614886731391592, 0.11992578849721706, 0.11084967864628882, 0.12436837328811888, 0.14461479703186381, 0.14534501716369877, 0.13330459594726563, 0.132965087890625, 0.13730288564685689, 0.1242600802456905, 0.12555821736653647, 0.14438756306966147, 0.13209152221679688, 0.10371788694174634, 0.10778204898980868, 0.13400590551181102, 0.14424342105263158, 0.13438795068379611, 0.14040613344384584, 0.14402584956485703, 0.13339493590614668, 0.15501971143531432, 0.13172022501627603, 0.13522782128298649, 0.12561346813988414, 0.14981361295859258, 0.13732635164127868, 0.1429581903025102, 0.1132264254385965, 0.11673177083333333, 0.1135656524122807, 0.12027052494517544, 0.12449801260964913, 0.13038041729242653, 0.13616981334477579, 0.13542895837803046, 0.14122505900021454, 0.13872090216691696, 0.10164524028160392, 0.10346637740796673, 0.10723875920055261, 0.13498980905385111, 0.095307624374448036, 0.13036786469344608, 0.099390243902439029, 0.12120906980050229, 0.11112663462613166, 0.12047553093259464, 0.098252971685237112, 0.10989364554473144, 0.10686058314956771, 0.15365868283991227, 0.10917434097837758, 0.11984022733906832, 0.11747476314008572, 0.09651290009447408, 0.12494570623922882, 0.097976502848497737, 0.10780428959905131, 0.086081102514847643, 0.083673615588509206, 0.13098718086247588, 0.12162082355199244, 0.11513155409845972, 0.10390021836800217, 0.13015237685686404, 0.12984069942072515, 0.11877145462347136, 0.13422884037760138, 0.13514401416005148, 0.1318118429521562, 0.0995311936530833, 0.11906607275426874, 0.098923740636278165, 0.12361985347229389, 0.12879308064778647, 0.098071361764800746, 0.099400087106790116, 0.11246963825724879, 0.12189563675787266, 0.11657270577435416, 0.1238105394408715, 0.1205092904302037, 0.13433456420898438, 0.14022318522135416, 0.13397725423177084, 0.13491058349609375, 0.13817596435546875, 0.1470184326171875, 0.12735493977864584, 0.13587443033854166, 0.13272984822591147, 0.13988876342773438, 0.13396708170572916, 0.1309814453125, 0.1322174072265625, 0.13536453247070313, 0.13395055135091147, 0.14148712158203125, 0.12568155924479166, 0.12782541910807291, 0.13590494791666666, 0.13919194539388022, 0.14190165201822916, 0.13313166300455728, 0.13582738240559897, 0.10481582300474922, 0.097972058425432246, 0.13780593872070313, 0.13452402750651041, 0.12846501668294272, 0.12778981526692709, 0.13025792439778647, 0.14262517293294272, 0.14192708333333334, 0.14794413248697916, 0.13939921061197916, 0.13486353556315103, 0.13083775838216147, 0.1463623046875, 0.13649749755859375, 0.13356272379557291, 0.13013839721679688, 0.13736979166666666, 0.13646570841471353, 0.13351694742838541, 0.13826115926106772, 0.13724772135416666, 0.13687642415364584, 0.13928731282552084, 0.13449350992838541, 0.13419977823893228, 0.12894821166992188, 0.137603759765625, 0.13727569580078125, 0.13621775309244791, 0.13578414916992188, 0.13533910115559897, 0.13838704427083334, 0.13175201416015625, 0.12602488199869791, 0.13371912638346353, 0.13364410400390625, 0.12292327880859374, 0.14132308959960938, 0.12876052856445314, 0.13512802124023438, 0.13334274291992188, 0.13485590616861978, 0.13868586222330728, 0.13797378540039063, 0.14207967122395834, 0.13890457153320313, 0.13632074991861978, 0.13989766438802084, 0.1064054704362903, 0.13087320317528428, 0.096214511041009462, 0.13449702317099335, 0.13766828470285347, 0.13812825520833333, 0.061284584505180165, 0.13833658854166667, 0.13423889723235358, 0.12045497169634391, 0.12750080454838017, 0.13092168172200522, 0.12595600328947368, 0.12327987938596491, 0.12785344709429824, 0.14424342105263158, 0.14040613344384584, 0.14402584956485703, 0.13253978724492679, 0.11428733159741877, 0.14143117268880209, 0.12549101967078913, 0.13732635164127868, 0.1429581903025102, 0.099748160911922595, 0.13872090216691696, 0.15365868283991227, 0.097884681731930309, 0.10715079211746523, 0.13098718086247588, 0.11884135472370766, 0.10539430500759135, 0.096600435598781217, 0.14057806801115641, 0.13422884037760138, 0.14520757348208538, 0.11689546131337585, 0.1203787041218331, 0.10219511187710978, 0.081627714449468258, 0.088828004709718328, 0.10639864240972274, 0.10306854214477028, 0.099581499920424463, 0.13242085774739584, 0.11447759266434199, 0.1192891508455234, 0.10268264993250914, 0.10419261875987729, 0.10856590803665676, 0.10852924165617664, 0.10466925291953287, 0.10638828062126182, 0.11032834933310678, 0.10834997828918802, 0.12837921626984128, 0.10894210357130353, 0.096620078334364048, 0.097897949615851965, 0.094467414049311083, 0.12721633911132813, 0.092724095307105403, 0.10997158069108059, 0.10722144331502902, 0.11872125013755915, 0.12830607096354166, 0.09595831006274845, 0.10427350427350428, 0.094459307183430041, 0.10370629299776082, 0.10417874014365243, 0.10500601709472039, 0.10223106608779546, 0.10576769053141805, 0.098767791076218289, 0.095144974301953983, 0.12831370035807291, 0.1029014254321996, 0.10214770282421509, 0.084887215372289077, 0.087760879583909407, 0.094484506640011429, 0.11242930094401042, 0.11850865681966145, 0.11963907877604167, 0.13491439819335938, 0.11063766479492188, 0.1109021504720052, 0.11421457926432292, 0.1160443623860677, 0.12207565307617188, 0.14811070760091147, 0.12186686197916667, 0.12248611450195313, 0.12305704752604167, 0.12493133544921875, 0.11210250854492188, 0.1064453125, 0.13931528727213541, 0.12932713826497397, 0.12727864583333334, 0.11992645263671875, 0.12623977661132813, 0.12243398030598958, 0.12103907267252605, 0.13272984822591147, 0.14280319213867188, 0.12239710489908855, 0.12392807006835938, 0.12585703531901041, 0.12497711181640625, 0.12796147664388022, 0.11536153157552083, 0.12370045979817708, 0.1114819844563802, 0.1219037373860677, 0.1151288350423177, 0.12504959106445313, 0.11812591552734375, 0.11700948079427083, 0.1172027587890625, 0.12329610188802083, 0.11154429117838542, 0.11405563354492188, 0.11887232462565105, 0.12135950724283855, 0.12349955240885417, 0.1161041259765625, 0.12960179646809897, 0.1185925801595052, 0.12484486897786458, 0.12619654337565103, 0.10786056518554688, 0.10869852701822917, 0.11479949951171875, 0.13010660807291666, 0.1054064432779948, 0.12585830688476563, 0.1173235575358073, 0.13548787434895834, 0.13209915161132813, 0.12721633911132813, 0.12648264567057291, 0.13515853881835938, 0.12812932332356772, 0.1046282450358073, 0.1085205078125, 0.10896555582682292, 0.10898717244466145, 0.1025848388671875, 0.10770924886067708, 0.11910374959309895, 0.09803009033203125, 0.113006591796875, 0.12870915730794272, 0.12527338663736978, 0.12001419067382813, 0.11152140299479167, 0.12222544352213542, 0.11556625366210938, 0.11044947306315105, 0.10848744710286458, 0.1241747538248698, 0.11540730794270833, 0.12067540486653645, 0.11911265055338542, 0.1304779052734375, 0.126129150390625, 0.10442795698924731, 0.10597476178212722, 0.10741069205875031, 0.10562167615241713, 0.11083453028560039, 0.094124964425716301, 0.13793818155924478, 0.10938207779419006, 0.13880284627278647, 0.10909939777098164, 0.11932793327452081, 0.097096347339774805, 0.11524905388838548, 0.1133988730627386, 0.11693217134288136, 0.11914467679030785, 0.096860460475052534, 0.13499425145766608, 0.11015801834160441, 0.13566462198893228, 0.10897213510588105, 0.10594108611186411, 0.093235647490778936, 0.097891566265060237, 0.12347631360236402, 0.11764244897959183, 0.11096248949154389, 0.098604044972049498, 0.090617554253332469, 0.10572494352112922, 0.10201225344543953, 0.08112432387794706, 0.089857088091863019, 0.09152947352225109, 0.08112432387794706, 0.10645139404673484, 0.093247807375932884, 0.099529900818837447, 0.11736145104316879, 0.12291317765770321, 0.11096991304552462, 0.11805237800446801, 0.10713038111761898, 0.10313319731856602, 0.094805292942184011, 0.10433715220949263, 0.094671201814058956, 0.098084998690097389, 0.10205652017425192, 0.075278469049594107, 0.11682619886370375, 0.097839689867614654, 0.10171839504968433, 0.12721633911132813, 0.099808562478245738, 0.10627862526596704, 0.085576082759181354, 0.1012694890510949, 0.088314199231617332, 0.085198289842252695, 0.086126496337323569, 0.102694636246297, 0.10065944785874549, 0.10672939388521366, 0.1073905862923204, 0.10700098149328638, 0.11193874385000664, 0.11140164655931238, 0.11228851611049248, 0.11434192885623248, 0.10971229595713777, 0.10619828605200946, 0.10999526689761109, 0.10747906589691579, 0.10018858470368515, 0.094333873900437362, 0.10602731276683917, 0.1018028391564898, 0.10673120776151582, 0.094312114546521275, 0.11612965269636577, 0.1023692699104993, 0.10443250115580213, 0.086176522709030451, 0.092049490374732298, 0.1003359687169154, 0.11406004984318996, 0.10586623701528254, 0.10815626021575678, 0.11344666303363431, 0.089017793352158675, 0.1037724257735486, 0.093701754811384041, 0.09918124006359301, 0.10650811702791901, 0.12507957728545963, 0.12261730387907413, 0.12891006469726563, 0.10315896188158961, 0.11320832923953625, 0.097650670699987674, 0.10601155066578716, 0.11528693915564901, 0.10314803213075766, 0.095751759292674127, 0.106060551502625, 0.10025917065390749, 0.10644628099173553, 0.11956575682382134, 0.10590822218334028, 0.12026875350751387, 0.11325992651369274, 0.11362172519846062, 0.11421728395061728, 0.11970802919708029, 0.11180864553314121, 0.10280743583576465, 0.10911490312457361, 0.10300659610369689, 0.10946133307002712, 0.091106243154435926, 0.10894876912840985, 0.10516965665524423, 0.096728267783427038, 0.10927066079111583, 0.10679534276519521, 0.10231169638449746, 0.10563284233497, 0.11004234724742892, 0.11124160739828635, 0.1072158486352096, 0.11149038122753858, 0.1040752051918305, 0.10572261168037321, 0.10012017893307813, 0.10159759954586003, 0.099129144191312565, 0.1074752814626872, 0.10261112977270183, 0.10458097200755624, 0.11156522162214762, 0.11861856620213924, 0.10359444477091535, 0.11546455220426226, 0.11163070265071828, 0.10202015881841381, 0.10549089459857339, 0.10878326320156463, 0.10235217963501991, 0.10106417298617952, 0.10051124744376277, 0.094367223950233281, 0.10671011265391669, 0.10035372067227427, 0.095687704602349322, 0.10145895504713183, 0.098343363345618148, 0.12512810002049601, 0.13092154631938224, 0.10606706574322208, 0.10520602978033318, 0.11442848510203132, 0.12598759174987958, 0.11427771556550952, 0.10205765244900185, 0.11553754396696743, 0.091431339977851606, 0.10540618363747616]

#Preparando o dataset
number_feats = 3
X = np.ones((len(listaLabels), number_feats))
y = np.ones((len(listaLabels)))
for i in range(811):
    y[i] = listaLabels[i]
    X[i,0] = listaQntRegions[i]
    X[i,1] = listaQntBlobs[i]
    X[i,2] = listaDensidade[i]

#Selecionando conjunto de validação (10% das amostras) -> 81 amostras
validation_X = np.ones((81, number_feats))
validation_y = np.ones((81))
#primeiro as amostras com glomerulopatias (51 amostras)
validation_X[0:51,:] = X[0:51,:]
validation_y[0:51] = y[0:51]
#Por fim, as amostras saudáveis (30 amostras)
validation_X[51:81,:] = X[781:811,:]
validation_y[51:81] = y[781:811]

#Separando o conjunto pro CROSS VALIDATION (90% restante)
X = X[51:781,:]
y = y[51:781]

fold = 10

n_neighbors=11
neigh = KNeighborsClassifier(n_neighbors, weights='uniform', algorithm='auto',  p=1) #KNN = 9 -> Melhor resultado
k_fold = cross_validation.StratifiedKFold(y, fold) #Cross Validation K-fold = 10 and KNN
accFinal = 0
prec = 0
rec = 0
print "=========================================="
print "KNN + Cross Validation 10-fold"
print "=========================================="
for k, (train, test) in enumerate(k_fold):
    neigh.fit(X[train], y[train])
    acc = neigh.score(X[test], y[test])
    y_predict = neigh.predict(X[test])
    print("fold",k, "-> score:", acc )
    #print("-> precision:", metrics.precision_score(y[test], y_predict))
    #print("-> recall:", metrics.recall_score(y[test], y_predict))
    accFinal += acc
    prec +=  metrics.precision_score(y[test], y_predict)
    rec += metrics.recall_score(y[test], y_predict)
      
accFinal = accFinal/(k+1); accfim = accFinal;
rec = rec/(k+1)
prec = prec/(k+1)
error_rate = 1 - accFinal

print "KNN Classification: k=",n_neighbors
print "Generalization Set: Accuracy =", accfim ,"%"
print "Precision:", prec
print "Recall:",rec
print "========================================================================"
neigh.fit(X, y) #New training with generalization set
print "Validation Set: Accuracy =", neigh.score(validation_X, validation_y) ,"%" 
y_predict = neigh.predict(validation_X)
print("Precision:", metrics.precision_score(validation_y, y_predict))
print("Recall:", metrics.recall_score(validation_y, y_predict))